mod arch;
mod loader;

use arch::PolicyNetwork;
use common::{LocalSettings, LRSchedule, Steps};
use loader::DataLoader;

use tch::{nn::{self, Optimizer, OptimizerConfig}, Device, Kind, Tensor};

impl common::Network for PolicyNetwork {
    type Inputs = (Tensor, Tensor, Tensor, usize);

    fn run_batch(
        &self,
        opt: &mut Optimizer,
        (xs, legal_mask, targets, batch_size): &Self::Inputs,
    ) -> f32 {
        let raw_logits = self.forward_raw(xs, *batch_size as i64);

        let masked = raw_logits.masked_fill(legal_mask, f64::NEG_INFINITY);

        let log_softmaxed = masked.log_softmax(1, Kind::Float);

        let masked_softmaxed = log_softmaxed.masked_fill(legal_mask, 0.0);

        let losses = -targets * masked_softmaxed;

        let loss = losses.sum(Kind::Float).divide_scalar(*batch_size as f64);

        opt.backward_step(&loss);

        tch::no_grad(|| f32::try_from(loss).unwrap())
    }

    fn save(&self, _: &str) {}
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let buffer_size_mb = args.next().unwrap().parse().unwrap();

    let steps = Steps {
        batch_size: 16_384,
        batches_per_superbatch: 1024,
        superbatches: 32,
    };

    let lr_schedule = LRSchedule {
        start: 0.01,
        gamma: 0.1,
        step: 14,
    };

    let local_settings = LocalSettings {
        output_path: "checkpoints/policy",
        data_path: "data/policygen6.binpack",
        save_rate: 10,
        print_rate: 16,
        buffer_size_mb,
    };

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = PolicyNetwork::randomised(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, lr_schedule.start.into()).unwrap();

    common::train::<PolicyNetwork, DataLoader>(
        device,
        &net,
        &mut opt,
        steps,
        lr_schedule,
        local_settings,
    )
    .unwrap();
}
