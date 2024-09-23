use std::{fs::File, io::BufReader};

use montyformat::{
    chess::{Move, Position},
    MontyFormat,
};
use tch::{Device, Kind, Tensor};

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

        let mut preallocs = PreAllocs {
            batch_indices: Vec::with_capacity(self.batch_size * 32),
            feat_indices: Vec::with_capacity(self.batch_size * 32),
            output_batch_indices: Vec::with_capacity(self.batch_size * 256),
            output_indices: Vec::with_capacity(self.batch_size * 256),
            output_values: Vec::with_capacity(self.batch_size * 256),
            mask_indices: Vec::with_capacity(self.batch_size * 256),
        };

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
                        let (xs, legal_mask, targets) = get_tensors(batch, &mut preallocs);

                        let xs = &xs.to_device(self.device).to_dense(None, true);
                        let legal_mask = &legal_mask
                            .to_device(self.device)
                            .to_dense(None, true)
                            .logical_not();
                        let targets = &targets.to_device(self.device).to_dense(None, true);

                        let should_break = f(xs, legal_mask, targets, batch.len());

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

struct PreAllocs {
    batch_indices: Vec<i64>,
    feat_indices: Vec<i64>,
    output_batch_indices: Vec<i64>,
    output_indices: Vec<i64>,
    output_values: Vec<f32>,
    mask_indices: Vec<i64>,
}

fn get_tensors(batch: &[DecompressedData], preallocs: &mut PreAllocs) -> (Tensor, Tensor, Tensor) {
    let batch_size = batch.len();

    preallocs.batch_indices.clear();
    preallocs.feat_indices.clear();
    preallocs.output_batch_indices.clear();
    preallocs.output_indices.clear();
    preallocs.output_values.clear();
    preallocs.mask_indices.clear();

    for (i, point) in batch.iter().enumerate() {
        arch::map_policy_inputs(&point.pos, |feat| {
            preallocs.batch_indices.push(i as i64);
            preallocs.feat_indices.push(feat as i64);
        });

        let mut total = 0;
        for value in &point.moves[..point.num] {
            total += value.1;
        }

        let total = f32::from(total);

        for value in &point.moves[..point.num] {
            let mov = Move::from(value.0);
            let move_idx = arch::map_move_to_index(&point.pos, mov);

            preallocs.mask_indices.push(move_idx as i64);
            preallocs.output_batch_indices.push(i as i64);
            preallocs.output_indices.push(move_idx as i64);
            preallocs.output_values.push(f32::from(value.1) / total);
        }
    }

    let total_feats = preallocs.feat_indices.len();
    let values = Tensor::from_slice(&vec![1f32; total_feats]);
    let indices = Tensor::from_slice2(&[&preallocs.batch_indices, &preallocs.feat_indices]);
    let xs = Tensor::sparse_coo_tensor_indices_size(
        &indices,
        &values,
        [batch_size as i64, arch::INPUTS],
        (Kind::Float, Device::Cpu),
        true,
    );

    let total_moves = preallocs.output_batch_indices.len();
    let values = Tensor::from_slice(&vec![true; total_moves]);
    let indices = Tensor::from_slice2(&[&preallocs.output_batch_indices, &preallocs.mask_indices]);
    let legal_mask = Tensor::sparse_coo_tensor_indices_size(
        &indices,
        &values,
        [batch_size as i64, arch::OUTPUTS],
        (Kind::Float, Device::Cpu),
        false,
    );

    let values = Tensor::from_slice(&preallocs.output_values);
    let indices =
        Tensor::from_slice2(&[&preallocs.output_batch_indices, &preallocs.output_indices]);
    let targets = Tensor::sparse_coo_tensor_indices_size(
        &indices,
        &values,
        [batch_size as i64, arch::OUTPUTS],
        (Kind::Float, Device::Cpu),
        false,
    );

    (xs, legal_mask, targets)
}
