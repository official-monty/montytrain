use std::num::NonZeroUsize;

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

use crate::inputs::{MAX_ACTIVE_BASE, MAX_MOVES};

#[derive(Debug)]
pub struct SingleLayer {
    l0w: AnnotatedNode,
    l0b: AnnotatedNode,
    l1w: AnnotatedNode,
    l1b: AnnotatedNode,
    input: AnnotatedNode,
    moves: AnnotatedNode,
}

impl SingleLayer {
    pub fn new<'a>(
        l0: Affine<'a, CudaMarker>,
        l1: Affine<'a, CudaMarker>,
        input: GraphBuilderNode<'a, CudaMarker>,
        moves: GraphBuilderNode<'a, CudaMarker>,
    ) -> Self {
        Self {
            l0w: l0.weights.annotated_node(),
            l0b: l0.bias.annotated_node(),
            l1w: l1.weights.annotated_node(),
            l1b: l1.bias.annotated_node(),
            input: input.annotated_node(),
            moves: moves.annotated_node(),
        }
    }
}

impl<B: BackendMarker> GraphIROperation<B> for SingleLayer {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.l0w, self.l0b, self.l1w, self.l1b, self.input, self.moves]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        assert_eq!(self.l0w.shape.cols(), self.input.shape.rows());
        assert_eq!(self.l0w.shape.rows(), self.l1w.shape.cols());

        util::check_dense_eq(ir, &self.input, false)?;
        util::check_dense_eq(ir, &self.l0w, true)?;
        util::check_dense_eq(ir, &self.l0b, true)?;
        util::check_dense_eq(ir, &self.l1w, true)?;
        util::check_dense_eq(ir, &self.l1b, true)?;
        util::check_not_batched(ir, &self.l0w)?;
        util::check_not_batched(ir, &self.l0b)?;
        util::check_not_batched(ir, &self.l1w)?;
        util::check_not_batched(ir, &self.l1b)?;

        Ok(Shape::new(1, self.input.shape.cols()))
    }

    fn ancillary_buffers(&self, _ir: &GraphIR<B>) -> Result<Vec<(Shape, Option<NonZeroUsize>)>, GraphIRError> {
        Ok(vec![(Shape::new(self.l0w.shape.rows(), self.input.shape.cols()), None)])
    }
}

impl GraphIROperationCompilable<CudaMarker> for SingleLayer {
    fn forward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let l0w = NodeId::new(self.l0w.idx, NodeIdTy::Values);
        let l0b = NodeId::new(self.l0b.idx, NodeIdTy::Values);
        let l1w = NodeId::new(self.l1w.idx, NodeIdTy::Values);
        let l1b = NodeId::new(self.l1b.idx, NodeIdTy::Values);
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let moves = NodeId::new(self.moves.idx, NodeIdTy::Values);

        let hl_output = NodeId::new(output_node, NodeIdTy::Ancillary(0));
        let output = NodeId::new(output_node, NodeIdTy::Values);

        let mut func = GraphFunction::default();

        func.push(MaybeUpdateBatchSize { input, output });
        func.push(MaybeUpdateBatchSize { input, output: hl_output });
        func.push(SingleLayerFwd { l0w, l0b, l1w, l1b, input, moves, hl_output, output });

        func
    }

    fn backward_pass(&self, _node_info: &GraphIRNodeInfo, output_node: usize) -> GraphFunction<CudaDevice> {
        let l1w = NodeId::new(self.l1w.idx, NodeIdTy::Values);
        let input = NodeId::new(self.input.idx, NodeIdTy::Values);
        let moves = NodeId::new(self.moves.idx, NodeIdTy::Values);

        let l0wg = NodeId::new(self.l0w.idx, NodeIdTy::Gradients);
        let l0bg = NodeId::new(self.l0b.idx, NodeIdTy::Gradients);
        let l1wg = NodeId::new(self.l1w.idx, NodeIdTy::Gradients);
        let l1bg = NodeId::new(self.l1b.idx, NodeIdTy::Gradients);

        let hl_output = NodeId::new(output_node, NodeIdTy::Ancillary(0));
        let output_grad = NodeId::new(output_node, NodeIdTy::Gradients);

        let mut func = GraphFunction::default();

        func.push(SingleLayerBwd { l1w, input, moves, hl_output, output_grad, l0wg, l0bg, l1wg, l1bg });

        func
    }
}

#[derive(Debug)]
struct SingleLayerFwd {
    l0w: NodeId,
    l0b: NodeId,
    l1w: NodeId,
    l1b: NodeId,
    input: NodeId,
    moves: NodeId,
    hl_output: NodeId,
    output: NodeId,
}

impl GraphInstruction<CudaDevice> for SingleLayerFwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let input = graph.get(self.input)?;
        let input = input.sparse()?;
        let moves = graph.get(self.moves)?;
        let moves = moves.sparse()?;

        let l0w = graph.get(self.l0w)?;
        let l0w = l0w.dense()?;
        let l0b = graph.get(self.l0b)?;
        let l0b = l0b.dense()?;
        let l1w = graph.get(self.l1w)?;
        let l1w = l1w.dense()?;
        let l1b = graph.get(self.l1b)?;
        let l1b = l1b.dense()?;

        let mut hl_output = graph.get_mut(self.hl_output)?;
        let hl_output = hl_output.dense_mut()?;

        let mut output = graph.get_mut(self.output)?;
        let output = output.dense_mut()?;

        let rows = hl_output.single_size() / MAX_MOVES;
        let batch_size = hl_output.batch_size();
        let nnz = input.nnz / MAX_MOVES;
        assert_eq!(input.nnz, MAX_MOVES * MAX_ACTIVE_BASE);
        assert_eq!(MAX_MOVES, 64);

        if batch_size != hl_output.batch_size()
            || batch_size != output.batch_size()
            || batch_size != moves.batch_size()
            || l0w.batch_size().is_some()
            || l0b.batch_size().is_some()
            || l1w.batch_size().is_some()
            || l1b.batch_size().is_some()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let device = input.buf.device.clone();

        unsafe {
            let threads = (rows / 4).min(512) as u32;

            assert!(threads.is_power_of_two(), "hl size must be a power of 2");

            let func = device.get_custom_func_or_rtc("single_layer_fwd", || {
                let kernel = include_str!("single_layer/fwd.cu");
                format!("#define THREADS {threads}\n{kernel}")
            })?;

            let batch_size = batch_size.unwrap_or(1) as u32;
            let grid_dim = (64 * batch_size, 1, 1);
            let block_dim = (threads, 1, 1);
            let cfg = LaunchConfig { grid_dim, block_dim, shared_mem_bytes: 4 * (threads + nnz as u32) };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(rows as i32))
                .arg(&(nnz as i32))
                .arg(&l0w.buf.buf)
                .arg(&l0b.buf.buf)
                .arg(&l1w.buf.buf)
                .arg(&l1b.buf.buf)
                .arg(&input.buf.buf)
                .arg(&moves.buf.buf)
                .arg(&mut hl_output.buf.buf)
                .arg(&mut output.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct SingleLayerBwd {
    input: NodeId,
    moves: NodeId,
    hl_output: NodeId,
    output_grad: NodeId,
    l1w: NodeId,
    l0wg: NodeId,
    l0bg: NodeId,
    l1wg: NodeId,
    l1bg: NodeId,
}

impl GraphInstruction<CudaDevice> for SingleLayerBwd {
    fn execute(&self, graph: &Graph<CudaDevice>) -> Result<(), OperationError<CudaError>> {
        let output_grad = graph.get(self.output_grad)?;
        let output_grad = output_grad.dense()?;

        let hl_output = graph.get(self.hl_output)?;
        let hl_output = hl_output.dense()?;

        let input = graph.get(self.input)?;
        let input = input.sparse()?;

        let moves = graph.get(self.moves)?;
        let moves = moves.sparse()?;

        let l1w = graph.get(self.l1w)?;
        let l1w = l1w.dense()?;

        let mut l0wg = graph.get_mut(self.l0wg)?;
        let l0wg = l0wg.dense_mut()?;

        let mut l0bg = graph.get_mut(self.l0bg)?;
        let l0bg = l0bg.dense_mut()?;

        let mut l1wg = graph.get_mut(self.l1wg)?;
        let l1wg = l1wg.dense_mut()?;

        let mut l1bg = graph.get_mut(self.l1bg)?;
        let l1bg = l1bg.dense_mut()?;

        let rows = hl_output.single_size() / MAX_MOVES;
        let batch_size = hl_output.batch_size();

        let nnz = input.nnz / MAX_MOVES;
        assert_eq!(input.nnz, MAX_MOVES * MAX_ACTIVE_BASE);

        if batch_size != input.batch_size()
            || batch_size != hl_output.batch_size()
            || l1w.batch_size().is_some()
            || l0wg.batch_size().is_some()
            || l0bg.batch_size().is_some()
            || l1wg.batch_size().is_some()
            || l1bg.batch_size().is_some()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let device = output_grad.buf.device.clone();

        unsafe {
            let func = device
                .get_custom_func_or_rtc("single_layer_bwd", || include_str!("single_layer/bwd.cu").to_string())?;

            let threads = (rows / 4).min(1024) as u32;
            let batch_size = batch_size.unwrap_or(1) as u32;
            let grid_dim = (64 * batch_size, 1, 1);
            let shared_mem_bytes = 4 * (4 * threads + nnz as u32);
            let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes };

            device
                .stream()
                .launch_builder(&func)
                .arg(&(rows as i32))
                .arg(&(nnz as i32))
                .arg(&input.buf.buf)
                .arg(&moves.buf.buf)
                .arg(&l1w.buf.buf)
                .arg(&hl_output.buf.buf)
                .arg(&output_grad.buf.buf)
                .arg(&mut l0wg.buf.buf)
                .arg(&mut l0bg.buf.buf)
                .arg(&mut l1wg.buf.buf)
                .arg(&mut l1bg.buf.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}
