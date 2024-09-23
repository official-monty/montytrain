mod arch;
mod loader;
mod rng;

use arch::PolicyNetwork;
use common::{LRSchedule, Steps};
use loader::DataLoader;

use tch::{Kind, nn::Optimizer, Tensor};

impl common::Network for PolicyNetwork {
    type Inputs = (Tensor, Tensor, Tensor, usize);

    fn new(vs: &tch::nn::Path) -> Self {
        Self::randomised(vs)
    }

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

    common::train::<PolicyNetwork, DataLoader>(
        "checkpoints/policy",
        "data/policygen6.binpack",
        buffer_size_mb,
        steps,
        lr_schedule,
        10,
    ).unwrap();
}
