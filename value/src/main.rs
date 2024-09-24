mod arch;
mod loader;
mod save;

use arch::ValueNetwork;
use loader::DataLoader;

use common::{LRSchedule, LocalSettings, Steps};
use tch::{nn::{self, OptimizerConfig}, Device, Tensor};

impl common::Network for ValueNetwork {
    type Inputs = (Vec<Tensor>, Tensor, usize);

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
    tch::set_num_threads(1);

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

    let local_settings = LocalSettings {
        output_path: "checkpoints/value",
        data_path: "../binpacks/new-data.binpack",
        save_rate: 10,
        print_rate: 16,
        buffer_size_mb: 4096,
    };

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = ValueNetwork::randomised(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, lr_schedule.start.into()).unwrap();

    common::train::<ValueNetwork, DataLoader>(
        device,
        &net,
        &mut opt,
        steps,
        lr_schedule,
        local_settings,
    )
    .unwrap();

    net.run_sample_fens();
}
