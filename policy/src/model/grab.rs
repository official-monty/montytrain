use bullet_core::{
    device::OperationError,
    graph::{
        builder::{GraphBuilderNode, Shape},
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

use crate::inputs::MAX_MOVES;

#[derive(Debug)]
pub struct Grab {
    input: AnnotatedNode,
    indices: AnnotatedNode,
}

impl Grab {
    pub fn new<'a>(input: GraphBuilderNode<'a, CudaMarker>, indices: GraphBuilderNode<'a, CudaMarker>) -> Self {
        Self { input: input.annotated_node(), indices: indices.annotated_node() }
    }
}

impl<B: BackendMarker> GraphIROperation<B> for Grab {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.indices, self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        assert_eq!(self.input.shape, self.indices.shape);

        util::check_same_batching(ir, &[&self.indices, &self.input])?;
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.indices, false)?;

        Ok(Shape::new(MAX_MOVES, 1))
    }
}

impl GraphIROperationCompilable<CudaMarker> for Grab {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let indices = NodeId::new(self.indices.idx, NodeIdTy::Values);
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input, output });
        func.push(GrabFwd { indices, input, output });

        func
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let indices = NodeId::new(self.indices.idx, NodeIdTy::Values);
        let input_grad = NodeId::new(self.input.idx, NodeIdTy::Gradients);
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input: output_grad, output: input_grad });
        func.push(GrabBwd { indices, input_grad, output_grad });

        func
    }
}

#[derive(Debug)]
struct GrabFwd {
    input: NodeId,
    indices: NodeId,
    output: NodeId,
}

impl GraphInstruction<CudaDevice> for GrabFwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let input = graph.get(self.input)?;
        let input = input.dense()?;

        let indices = graph.get(self.indices)?;
        let indices = indices.sparse()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let single_size = input.single_size();
        let batch_size = input.batch_size();
        let nnz = indices.nnz;
        assert_eq!(nnz, MAX_MOVES);

        if batch_size != indices.batch_size() || batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if single_size != indices.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let device = input.buf.device.clone();

        unsafe {
            let func = device.get_custom_func_or_rtc("grab_fwd", || include_str!("grab_fwd.cu").to_string())?;

            let threads = 1024u32;
            let batch_size = batch_size.unwrap_or(1);
            let blocks = ((batch_size * nnz) as u32).div_ceil(threads);
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
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
struct GrabBwd {
    input_grad: NodeId,
    indices: NodeId,
    output_grad: NodeId,
}

impl GraphInstruction<CudaDevice> for GrabBwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let output_grad = graph.get(self.output_grad)?;
        let output_grad = output_grad.dense()?;

        let indices = graph.get(self.indices)?;
        let indices = indices.sparse()?;

        let mut input_grad = graph.get_mut(self.input_grad)?;
        let input_grad = input_grad.dense_mut()?;

        let single_size = input_grad.single_size();
        let batch_size = input_grad.batch_size();
        let nnz = indices.nnz;
        assert_eq!(nnz, MAX_MOVES);

        if batch_size != indices.batch_size() || batch_size != output_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if single_size != indices.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let device = output_grad.buf.device.clone();

        unsafe {
            let func = device.get_custom_func_or_rtc("grab_bwd", || include_str!("grab_bwd.cu").to_string())?;

            let threads = 1024u32;
            let batch_size = batch_size.unwrap_or(1);
            let blocks = ((batch_size * nnz) as u32).div_ceil(threads);
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
                .arg(&output_grad.buf.buf)
                .arg(&indices.buf.buf)
                .arg(&mut input_grad.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
