use bullet_core::{
    acyclib::graph::NodeId,
    device::OperationError,
    function::{DeviceFunction, DeviceOperation, MaybeUpdateBatchSize},
    graph::{
        builder::{Affine, GraphBuilderNode, Shape},
        ir::{
            node::AnnotatedNode,
            operation::{util, GraphIROperationBase, GraphIROperationCompilable},
            BackendMarker, GraphIR, GraphIRError,
        },
        Graph, GraphNodeIdTy,
    },
    tensor::TensorRef,
};
use bullet_cuda_backend::{
    cudarc::driver::{LaunchConfig, PushKernelArg},
    CudaDevice, CudaError, CudaMarker,
};

use crate::inputs::{MAX_MOVES, NUM_MOVES_INDICES};

#[derive(Debug)]
pub struct SelectAffine {
    weights: AnnotatedNode,
    biases: AnnotatedNode,
    input: AnnotatedNode,
    indices: AnnotatedNode,
}

impl SelectAffine {
    pub fn new<'a>(
        affine: Affine<'a, CudaMarker>,
        input: GraphBuilderNode<'a, CudaMarker>,
        indices: GraphBuilderNode<'a, CudaMarker>,
    ) -> Self {
        Self {
            weights: affine.weights.reshape(affine.weights.annotated_node().shape.transpose()).annotated_node(),
            biases: affine.bias.annotated_node(),
            input: input.annotated_node(),
            indices: indices.annotated_node(),
        }
    }
}

impl<B: BackendMarker> GraphIROperationBase<B> for SelectAffine {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.indices, self.input, self.weights, self.biases]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        assert_eq!(self.weights.shape, Shape::new(self.input.shape.rows(), NUM_MOVES_INDICES));
        assert_eq!(self.biases.shape, Shape::new(NUM_MOVES_INDICES, 1));

        util::check_same_batching(ir, &[&self.indices, &self.input])?;
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.indices, false)?;
        util::check_dense_eq(ir, &self.weights, true)?;
        util::check_dense_eq(ir, &self.biases, true)?;
        util::check_not_batched(ir, &self.weights)?;
        util::check_not_batched(ir, &self.biases)?;

        Ok(Shape::new(MAX_MOVES, 1))
    }
}

impl GraphIROperationCompilable<CudaMarker> for SelectAffine {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);
        let weights = graph.get_ref(self.weights.idx, GraphNodeIdTy::Values);
        let biases = graph.get_ref(self.biases.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(SelectAffineFwd { indices, input, weights, biases, output });

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let indices = graph.get_ref(self.indices.idx, GraphNodeIdTy::Values);
        let weights = graph.get_ref(self.weights.idx, GraphNodeIdTy::Values);
        let input_grad = graph.get_ref(self.input.idx, GraphNodeIdTy::Gradients);
        let weights_grad = graph.get_ref(self.weights.idx, GraphNodeIdTy::Gradients);
        let biases_grad = graph.get_ref(self.biases.idx, GraphNodeIdTy::Gradients);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let mut func = DeviceFunction::default();

        func.push(MaybeUpdateBatchSize { input: output_grad.clone(), output: input_grad.clone() });
        func.push(SelectAffineBwd { input, weights, indices, input_grad, weights_grad, biases_grad, output_grad });

        func
    }
}

#[derive(Debug)]
struct SelectAffineFwd {
    weights: TensorRef<CudaDevice>,
    biases: TensorRef<CudaDevice>,
    input: TensorRef<CudaDevice>,
    indices: TensorRef<CudaDevice>,
    output: TensorRef<CudaDevice>,
}

impl DeviceOperation<CudaDevice> for SelectAffineFwd {
    fn opname(&self) -> String {
        "SelectAffineFwd".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<CudaError>> {
        let input = self.input.dense();
        let weights = self.weights.dense();
        let biases = self.biases.dense();
        let indices = self.indices.sparse();
        let mut output = self.output.dense_mut();

        let single_size = input.single_size();
        let batch_size = input.batch_size();
        let nnz = indices.nnz;
        assert_eq!(nnz, MAX_MOVES);
        assert_eq!(nnz, 64);

        if batch_size != indices.batch_size()
            || batch_size != output.batch_size()
            || weights.batch_size().is_some()
            || biases.batch_size().is_some()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let device = input.buf.device.clone();

        unsafe {
            let threads = (single_size / 4).min(512) as u32;

            assert!(threads.is_power_of_two(), "hl size must be a power of 2");

            let func = device.get_custom_func_or_rtc("select_affine_fwd", || {
                let kernel = include_str!("select_affine/fwd.cu");
                format!("#define THREADS {threads}\n{kernel}")
            })?;

            let batch_size = batch_size.unwrap_or(1) as u32;
            let grid_dim = (64, batch_size, 1);
            let block_dim = (threads, 1, 1);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 4 * threads };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
                .arg(&weights.buf.buf)
                .arg(&biases.buf.buf)
                .arg(&input.buf.buf)
                .arg(&indices.buf.buf)
                .arg(&mut output.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct SelectAffineBwd {
    input: TensorRef<CudaDevice>,
    weights: TensorRef<CudaDevice>,
    indices: TensorRef<CudaDevice>,
    output_grad: TensorRef<CudaDevice>,
    input_grad: TensorRef<CudaDevice>,
    weights_grad: TensorRef<CudaDevice>,
    biases_grad: TensorRef<CudaDevice>,
}

impl DeviceOperation<CudaDevice> for SelectAffineBwd {
    fn opname(&self) -> String {
        "SelectAffineBwd".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<CudaError>> {
        let output_grad = self.output_grad.dense();
        let indices = self.indices.sparse();
        let input = self.input.dense();
        let weights = self.weights.dense();
        let mut input_grad = self.input_grad.dense_mut();
        let mut weights_grad = self.weights_grad.dense_mut();
        let mut biases_grad = self.biases_grad.dense_mut();

        let single_size = input_grad.single_size();
        let batch_size = input_grad.batch_size();
        let nnz = indices.nnz;
        assert_eq!(nnz, MAX_MOVES);

        if batch_size != indices.batch_size()
            || batch_size != output_grad.batch_size()
            || batch_size != input.batch_size()
            || batch_size != input_grad.batch_size()
            || weights.batch_size().is_some()
            || weights_grad.batch_size().is_some()
            || biases_grad.batch_size().is_some()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let device = output_grad.buf.device.clone();

        unsafe {
            let func = device
                .get_custom_func_or_rtc("select_affine_bwd", || include_str!("select_affine/bwd.cu").to_string())?;

            let threads = (single_size / 4).min(1024) as u32;
            let batch_size = batch_size.unwrap_or(1) as u32;
            let grid_dim = (64, batch_size, 1);
            let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 16 * threads };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
                .arg(&weights.buf.buf)
                .arg(&input.buf.buf)
                .arg(&indices.buf.buf)
                .arg(&output_grad.buf.buf)
                .arg(&mut input_grad.buf.buf)
                .arg(&mut weights_grad.buf.buf)
                .arg(&mut biases_grad.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
