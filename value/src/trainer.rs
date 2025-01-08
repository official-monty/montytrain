use bullet::{
    optimiser::{AdamWOptimiser, Optimiser},
    NetworkTrainer,
};

use crate::preparer::PreparedData;

pub struct Trainer {
    pub optimiser: AdamWOptimiser,
}

impl NetworkTrainer for Trainer {
    type PreparedData = PreparedData;
    type Optimiser = AdamWOptimiser;

    fn optimiser(&self) -> &Self::Optimiser {
        &self.optimiser
    }

    fn optimiser_mut(&mut self) -> &mut Self::Optimiser {
        &mut self.optimiser
    }

    fn load_batch(&mut self, prepared: &Self::PreparedData) -> usize {
        let batch_size = prepared.batch_size;

        let graph = self.optimiser.graph_mut();

        let inputs = &prepared.inputs;
        unsafe {
            graph.get_input_mut("inputs").load_sparse_from_slice(
                inputs.shape,
                inputs.max_active,
                &inputs.value,
            );
        }

        let stm_mask = &prepared.stm_mask;
        unsafe {
            graph.get_input_mut("stm_mask").load_sparse_from_slice(
                stm_mask.shape,
                stm_mask.max_active,
                &stm_mask.value,
            );
        }

        let ntm_mask = &prepared.stm_mask;
        unsafe {
            graph.get_input_mut("ntm_mask").load_sparse_from_slice(
                ntm_mask.shape,
                ntm_mask.max_active,
                &ntm_mask.value,
            );
        }

        let target = &prepared.target;
        graph
            .get_input_mut("target")
            .load_dense_from_slice(target.shape, &target.value);

        batch_size
    }
}
