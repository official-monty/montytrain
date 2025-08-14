use bullet_core::{
    device::OperationError,
    graph::{
        builder::{Affine, GraphBuilderNode, Shape},
        instruction::{GraphInstruction, MaybeUpdateBatchSize},
        ir::{
            node::AnnotatedNode,
            operation::{util, GraphIROperation, GraphIROperationCompilable},
            BackendMarker, GraphIR, GraphIRError, GraphIRNodeInfo,
        },
        Graph, GraphFunction, NodeId, NodeIdTy,
    },
};
use bullet_cuda_backend::{CudaDevice, CudaError, CudaMarker};
use cudarc::driver::{LaunchConfig, PushKernelArg};

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

impl<B: BackendMarker> GraphIROperation<B> for SelectAffine {
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
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let indices = NodeId::new(self.indices.idx, NodeIdTy::Values);
        let weights = NodeId::new(self.weights.idx, NodeIdTy::Values);
        let biases = NodeId::new(self.biases.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input, output });
        func.push(SelectAffineFwd { indices, input, weights, biases, output });

        func
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let indices = NodeId::new(self.indices.idx, NodeIdTy::Values);
        let weights = NodeId::new(self.weights.idx, NodeIdTy::Values);

        let input_grad = NodeId::new(self.input.idx, NodeIdTy::Gradients);
        let weights_grad = NodeId::new(self.weights.idx, NodeIdTy::Gradients);
        let biases_grad = NodeId::new(self.biases.idx, NodeIdTy::Gradients);
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input: output_grad, output: input_grad });
        func.push(SelectAffineBwd { input, weights, indices, input_grad, weights_grad, biases_grad, output_grad });

        func
    }
}

#[derive(Debug)]
struct SelectAffineFwd {
    weights: NodeId,
    biases: NodeId,
    input: NodeId,
    indices: NodeId,
    output: NodeId,
}

impl GraphInstruction<CudaDevice> for SelectAffineFwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let weights = graph.get(self.weights)?;
        let weights = weights.dense()?;

        let biases = graph.get(self.biases)?;
        let biases = biases.dense()?;

        let indices = graph.get(self.indices)?;
        let indices = indices.sparse()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

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
            let grid_dim = (batch_size, 1, 1);
            let block_dim = (64, 1, 1);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 0 };

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
    input: NodeId,
    weights: NodeId,
    indices: NodeId,
    output_grad: NodeId,

    input_grad: NodeId,
    weights_grad: NodeId,
    biases_grad: NodeId,
}

impl GraphInstruction<CudaDevice> for SelectAffineBwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let output_grad = graph.get(self.output_grad)?;
        let output_grad = output_grad.dense()?;

        let indices = graph.get(self.indices)?;
        let indices = indices.sparse()?;

        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let weights = graph.get(self.weights)?;
        let weights = weights.dense()?;

        let mut input_grad = graph.get_mut(self.input_grad)?;
        let input_grad = input_grad.dense_mut()?;

        let mut weights_grad = graph.get_mut(self.weights_grad)?;
        let weights_grad = weights_grad.dense_mut()?;

        let mut biases_grad = graph.get_mut(self.biases_grad)?;
        let biases_grad = biases_grad.dense_mut()?;

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

            let batch_size = batch_size.unwrap_or(1) as u32;
            let grid_dim = (batch_size, 1, 1);
            let cfg = LaunchConfig { grid_dim, block_dim: (64, 1, 1), shared_mem_bytes: 0 };

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
