mod arch;
mod loader;
mod rng;

use arch::PolicyNetwork;
use loader::DataLoader;
use tch::{
    nn::{self, Optimizer, OptimizerConfig},
    Device, Kind, TchError, Tensor,
};

use std::{io::Write, time::Instant};

const BATCH_SIZE: usize = 16_384;
const BPSB: usize = 1024;

fn run_batch(
    net: &PolicyNetwork,
    opt: &mut Optimizer,
    xs: &Tensor,
    legal_mask: &Tensor,
    targets: &Tensor,
    batch_size: usize,
) -> f32 {
    let raw_logits = net.forward_raw(xs, batch_size as i64);

    let masked = raw_logits.masked_fill(legal_mask, f64::NEG_INFINITY);

    let log_softmaxed = masked.log_softmax(1, Kind::Float);

    let masked_softmaxed = log_softmaxed.masked_fill(legal_mask, 0.0);

    let losses = -targets * masked_softmaxed;

    let loss = losses.sum(Kind::Float).divide_scalar(batch_size as f64);

    opt.backward_step(&loss);

    tch::no_grad(|| f32::try_from(loss).unwrap())
}

pub fn train(
    buffer_size_mb: usize,
    data_path: String,
    superbatches: usize,
    lr_start: f32,
    lr_end: f32,
    final_lr_superbatch: usize,
) -> Result<(), TchError> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = PolicyNetwork::new(&vs.root());

    let mut lr = lr_start;

    let mut opt = nn::Adam::default().build(&vs, lr.into())?;

    let mut running_error = 0.0;
    let mut sb = 0;
    let mut batch_no = 0;

    let data_loader = DataLoader::new(data_path.as_str(), buffer_size_mb, BATCH_SIZE, device);

    let mut t = Instant::now();

    data_loader.map_batches(|xs, legal_mask, targets, batch_size| {
        let t2 = Instant::now();
        let this_loss = run_batch(&net, &mut opt, xs, legal_mask, targets, batch_size);

        running_error += this_loss;

        batch_no += 1;
        print!(
            "> Superbatch {}/{superbatches} Batch {}/{BPSB} Current Loss {this_loss:.4} Time {}ms\r",
            sb + 1,
            batch_no % BPSB,
            t2.elapsed().as_millis()
        );
        let _ = std::io::stdout().flush();

        if batch_no % BPSB == 0 {
            let elapsed = t.elapsed().as_secs_f32();
            t = Instant::now();

            sb += 1;
            println!(
                "> Superbatch {sb}/{superbatches} Running Loss {} Time {:.2}s",
                running_error / BPSB as f32,
                elapsed,
            );

            let mut seconds_left = ((superbatches - sb) as f32 * elapsed) as u64;
            let mut minutes_left = seconds_left / 60;
            seconds_left -= minutes_left * 60;
            let hours_left = minutes_left / 60;
            minutes_left -= hours_left * 60;

            println!("Estimated {hours_left}h {minutes_left}m {seconds_left}s Left in Training",);

            running_error = 0.0;

            let decay_factor = (lr_end / lr_start).powf(1.0 / final_lr_superbatch as f32);

            if sb >= final_lr_superbatch {
                lr = lr_end;
            } else {
                lr = lr_start * decay_factor.powf(sb as f32);
            }
            println!("Dropping LR to {lr}");
            opt.set_lr(lr.into());

            //if sb % 10 == 0 {
                vs.save(format!("checkpoints/policy-{sb}.pt").as_str()).unwrap();
            //}

            sb == superbatches
        } else {
            false
        }
    });

    Ok(())
}
