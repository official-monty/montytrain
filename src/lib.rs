mod arch;
mod loader;
mod rng;

use arch::PolicyNetwork;
use loader::DataLoader;
use tch::{nn::{self, OptimizerConfig}, Device, TchError, Tensor};

use std::{io::Write, time::Instant};

const BATCH_SIZE: usize = 16_384;
const BPSB: usize = 1024;

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

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?; 

    let mut lr = lr_start;

    let mut running_error = 0.0;
    let mut sb = 0;
    let mut batch_no = 0;

    let data_loader = DataLoader::new(data_path.as_str(), buffer_size_mb, BATCH_SIZE);

    let mut t = Instant::now();

    data_loader.map_batches(|xs, legal_mask, targets, batch_size| {
        let fwd = net.forward(&xs.to_device(device), &legal_mask.to_device(device), batch_size as i64);

        println!("forward");
        println!("{:?}", fwd.size());

        println!("loss");
        println!("{:?}", targets.size());
        let loss = fwd.cross_entropy_loss::<Tensor>(&targets.to_device(device), None, tch::Reduction::Mean, -100, 0.0);
        println!("{:?}", loss.size());

        println!("loss: {}", loss);

        println!("backprop");
        opt.backward_step(&loss);

        running_error += loss.double_value(&[0]);
        batch_no += 1;
        print!(
            "> Superbatch {}/{superbatches} Batch {}/{BPSB}\r",
            sb + 1,
            batch_no % BPSB,
        );
        let _ = std::io::stdout().flush();

        if batch_no % BPSB == 0 {
            let elapsed = t.elapsed().as_secs_f32();
            t = Instant::now();

            sb += 1;
            println!(
                "> Superbatch {sb}/{superbatches} Running Loss {} Time {:.2}s",
                running_error / (BPSB * BATCH_SIZE) as f64,
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

            if sb % 10 == 0 {
                vs.save(format!("checkpoints/policy-{sb}.pt").as_str()).unwrap();
            }

            sb == superbatches
        } else {
            false
        }
    });

    Ok(())
}
