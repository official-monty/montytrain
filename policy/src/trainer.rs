use bullet::{
    nn::optimiser::{AdamWOptimiser, Optimiser},
    ExecutionContext, NetworkTrainer,
};

use crate::preparer::PreparedData;

pub struct Trainer {
    pub optimiser: Optimiser<ExecutionContext, AdamWOptimiser>,
}

impl NetworkTrainer for Trainer {
    type PreparedData = PreparedData;
    type OptimiserState = AdamWOptimiser;

    fn optimiser(&self) -> &Optimiser<ExecutionContext, Self::OptimiserState> {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Optimiser<ExecutionContext, Self::OptimiserState> {
        &mut self.optimiser
    }

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = &mut self.optimiser.graph;

        let inputs = &prepared.inputs;
        unsafe {
            graph
                .get_input_mut("inputs")
                .load_sparse_from_slice(inputs.max_active, Some(batch_size), &inputs.value)
                .unwrap();
        }

        let mask = &prepared.mask;
        unsafe {
            graph
                .get_input_mut("mask")
                .load_sparse_from_slice(mask.max_active, Some(batch_size), &mask.value)
                .unwrap();
        }

        let dist = &prepared.dist;
        graph
            .get_input_mut("dist")
            .load_dense_from_slice(Some(batch_size), &dist.value)
            .unwrap();

        batch_size
    }
}
