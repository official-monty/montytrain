use std::{fs::File, io::BufReader};

use montyformat::{chess::{Move, Position}, MontyFormat};
use tch::{Device, Tensor};

use crate::{arch, rng::Rand};

#[derive(Clone, Copy)]
pub struct DecompressedData {
    pub pos: Position,
    pub moves: [(u16, u16); 112],
    pub num: usize,
}

pub struct DataLoader {
    file_path: String,
    buffer_size: usize,
    batch_size: usize,
    device: Device,
}

impl DataLoader {
    pub fn new(path: &str, buffer_size_mb: usize, batch_size: usize, device: Device) -> Self {
        Self {
            file_path: path.to_string(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<DecompressedData>() / 2,
            batch_size,
            device,
        }
    }

    pub fn map_batches<F: FnMut(&Tensor, &Tensor, &Tensor, usize) -> bool>(&self, mut f: F) {
        let mut reusable_buffer = Vec::new();

        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        'dataloading: loop {
            let mut reader = BufReader::new(File::open(self.file_path.as_str()).unwrap());

            while let Ok(game) = MontyFormat::deserialise_from(&mut reader) {
                parse_into_buffer(game, &mut reusable_buffer);

                if shuffle_buffer.len() + reusable_buffer.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&reusable_buffer);
                } else {
                    println!("#[Shuffling]");
                    shuffle(&mut shuffle_buffer);

                    println!("#[Running Batches]");
                    for batch in shuffle_buffer.chunks(self.batch_size) {
                        let (xs, legal_mask, targets) = get_tensors(batch);
                        
                        let should_break = f(
                            &xs.to_device(self.device),
                            &legal_mask.to_device(self.device),
                            &targets.to_device(self.device),
                            batch.len(),
                        );

                        if should_break {
                            break 'dataloading;
                        }
                    }

                    println!();
                    shuffle_buffer.clear();
                }
            }
        }
    }
}

fn shuffle(data: &mut [DecompressedData]) {
    let mut rng = Rand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

fn parse_into_buffer(game: MontyFormat, buffer: &mut Vec<DecompressedData>) {
    buffer.clear();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        if (data.score - 0.5).abs() > 0.49 {
        } else if let Some(dist) = data.visit_distribution.as_ref() {
            if dist.len() < 112 {
                let mut data_point = DecompressedData {
                    pos,
                    moves: [(0, 0); 112],
                    num: dist.len(),
                };

                for (i, (mov, visits)) in dist.iter().enumerate() {
                    data_point.moves[i] = (u16::from(*mov), *visits as u16);
                }

                buffer.push(data_point);
            }
        }

        pos.make(data.best_move, &castling);
    }
}

pub fn get_tensors(batch: &[DecompressedData]) -> (Tensor, Tensor, Tensor) {
    let batch_size = batch.len();

    let mut inputs = vec![0f32; batch_size * arch::INPUTS as usize];
    let mut masks = vec![true; batch_size * arch::OUTPUTS as usize];
    let mut outputs = vec![0f32; batch_size * arch::OUTPUTS as usize];

    for (i, point) in batch.iter().enumerate() {
        arch::map_policy_inputs(&point.pos, |feat| inputs[arch::INPUTS as usize * i + feat] = 1.0);

        let mut total = 0;
        for value in &point.moves[..point.num] {
            total += value.1;
        }

        let total = f32::from(total);

        for value in &point.moves[..point.num] {
            let mov = Move::from(value.0);
            let move_idx = arch::map_move_to_index(&point.pos, mov);
            let idx = arch::OUTPUTS as usize * i + move_idx;
            
            masks[idx] = false;
            outputs[idx] += f32::from(value.1) / total;
        }
    }

    let xs = Tensor::from_slice(&inputs).reshape([batch_size as i64, arch::INPUTS]);
    let legal_mask = Tensor::from_slice(&masks).reshape([batch_size as i64, arch::OUTPUTS]);
    let targets = Tensor::from_slice(&outputs).reshape([batch_size as i64, arch::OUTPUTS]);

    (xs, legal_mask, targets)
}
