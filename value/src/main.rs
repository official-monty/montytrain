mod arch;
mod loader;
mod save;

use arch::ValueNetwork;
use loader::DataLoader;

use common::{LRSchedule, Steps};
use tch::Tensor;

impl common::Network for ValueNetwork {
    type Inputs = (Vec<Tensor>, Tensor, usize);

    fn new(vs: &tch::nn::Path) -> Self {
        Self::randomised(vs)
    }

    fn run_batch(&self, opt: &mut tch::nn::Optimizer, (xs, targets, batch_size): &Self::Inputs) -> f32 {
        let out = self.fwd(xs, *batch_size as i64).reshape([*batch_size as i64]);
        
        let predicted = out.sigmoid();

        let loss = predicted.mse_loss(targets, tch::Reduction::Mean);

        opt.backward_step(&loss);

        tch::no_grad(|| f32::try_from(loss).unwrap())
    }

    fn save(&self, path: &str) {
        let net = self.export();
        net.write_to_bin(path);
    }
}

fn main() {
    let steps = Steps {
        batch_size: 16_384,
        batches_per_superbatch: 6104,
        superbatches: 40,
    };

    let lr_schedule = LRSchedule {
        start: 0.001,
        gamma: 0.1,
        step: 18,
    };

    common::train::<ValueNetwork, DataLoader>(
        "checkpoints/value",
        "../binpacks/new-data.binpack",
        1024,
        steps,
        lr_schedule,
        10,
        16,
    )
    .unwrap();
}
