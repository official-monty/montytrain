mod rng;

pub use rng::Rand;

use std::{io::Write, time::Instant};

use tch::{
    nn::Optimizer,
    Device, TchError,
};

pub trait Network {
    type Inputs;

    fn run_batch(&self, opt: &mut Optimizer, inputs: &Self::Inputs) -> f32;

    fn save(&self, path: &str);
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

#[derive(Clone, Copy, Debug)]
pub struct LocalSettings {
    pub output_path: &'static str,
    pub data_path: &'static str,
    pub save_rate: usize,
    pub print_rate: usize,
    pub buffer_size_mb: usize,
}

pub fn train<N: Network, D: DataLoader<N>>(
    device: Device,
    net: &N,
    opt: &mut Optimizer,
    steps: Steps,
    lr_schedule: LRSchedule,
    local_settings: LocalSettings,
) -> Result<(), TchError> {
    println!("Device: {device:?}");

    let data_loader = D::new(
        local_settings.data_path,
        local_settings.buffer_size_mb,
        steps.batch_size,
        device,
    );

    let bpsb = steps.batches_per_superbatch;
    let sbs = steps.superbatches;

    let mut lr = lr_schedule.start;

    let mut running_error = 0.0;
    let mut sb = 0;
    let mut batch_no = 0;

    let mut t = Instant::now();

    let mut window_time = 0;

    data_loader.map_batches(|inputs| {
        let t2 = Instant::now();
        let this_loss = net.run_batch(opt, inputs);
        window_time += t2.elapsed().as_millis();
        running_error += this_loss;

        batch_no += 1;
        if batch_no % local_settings.print_rate == 0 {
            print!("\r\x1b[K");
            print!(
                "> Superbatch {}/{sbs} Batch {}/{bpsb} Current Loss {this_loss:.4} Time {}ms",
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

            print!("\r\x1b[K");
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

            if sb % local_settings.save_rate == 0 {
                net.save(format!("{}/network-{sb}.network", local_settings.output_path).as_str())
            }

            sb == sbs
        } else {
            false
        }
    });

    Ok(())
}
