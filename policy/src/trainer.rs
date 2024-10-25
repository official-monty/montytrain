use bullet::{optimiser::{AdamWOptimiser, Optimiser}, NetworkTrainer};

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
            graph.get_input_mut("inputs").load_sparse_from_slice(inputs.shape, inputs.max_active, &inputs.value);
        }

        let dist = &prepared.dist;
        graph.get_input_mut("dist").load_dense_from_slice(dist.shape, &dist.value);

        batch_size
    }
}
