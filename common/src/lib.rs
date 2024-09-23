mod rng;

pub use rng::Rand;

use std::{io::Write, time::Instant};

use tch::{
    nn::{self, Optimizer, OptimizerConfig},
    Device, TchError,
};

pub trait Network {
    type Inputs;

    fn new(vs: &nn::Path) -> Self;

    fn run_batch(&self, opt: &mut Optimizer, inputs: &Self::Inputs) -> f32;
}

pub trait DataLoader<N: Network> {
    fn new(path: &str, buffer_size_mb: usize, batch_size: usize, device: Device) -> Self;

    fn map_batches<F: FnMut(&N::Inputs) -> bool>(&self, f: F);
}

#[derive(Clone, Copy, Debug)]
pub struct Steps {
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub superbatches: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct LRSchedule {
    pub start: f32,
    pub gamma: f32,
    pub step: usize,
}

pub fn train<N: Network, D: DataLoader<N>>(
    output_path: &str,
    data_path: &str,
    buffer_size_mb: usize,
    steps: Steps,
    lr_schedule: LRSchedule,
    save_rate: usize,
    print_rate: usize,
) -> Result<(), TchError> {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = N::new(&vs.root());

    let data_loader = D::new(data_path, buffer_size_mb, steps.batch_size, device);

    let bpsb = steps.batches_per_superbatch;
    let sbs = steps.superbatches;

    let mut lr = lr_schedule.start;

    let mut opt = nn::Adam::default().build(&vs, lr.into())?;

    let mut running_error = 0.0;
    let mut sb = 0;
    let mut batch_no = 0;

    let mut t = Instant::now();

    let mut window_time = 0;

    data_loader.map_batches(|inputs| {
        let t2 = Instant::now();
        let this_loss = net.run_batch(&mut opt, inputs);
        window_time += t2.elapsed().as_millis();
        running_error += this_loss;

        batch_no += 1;
        if batch_no % print_rate == 0 {
            print!(
                "> Superbatch {}/{sbs} Batch {}/{bpsb} Current Loss {this_loss:.4} Time {}ms\r",
                sb + 1,
                batch_no % bpsb,
                window_time
            );
            let _ = std::io::stdout().flush();
            window_time = 0;
        }

        if batch_no % bpsb == 0 {
            let elapsed = t.elapsed().as_secs_f32();
            t = Instant::now();

            sb += 1;
            println!(
                "> Superbatch {sb}/{sbs} Running Loss {} Time {:.2}s",
                running_error / bpsb as f32,
                elapsed,
            );

            let mut seconds_left = ((sbs - sb) as f32 * elapsed) as u64;
            let mut minutes_left = seconds_left / 60;
            seconds_left -= minutes_left * 60;
            let hours_left = minutes_left / 60;
            minutes_left -= hours_left * 60;

            println!("Estimated {hours_left}h {minutes_left}m {seconds_left}s Left in Training",);

            running_error = 0.0;

            if sb % lr_schedule.step == 0 {
                lr *= lr_schedule.gamma;
                println!("Dropping LR to {lr}");
            }

            opt.set_lr(lr.into());

            if sb % save_rate == 0 {
                vs.save(format!("{output_path}/network-{sb}.pt").as_str())
                    .unwrap();
            }

            sb == sbs
        } else {
            false
        }
    });

    Ok(())
}
