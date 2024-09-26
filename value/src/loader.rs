use std::{fs::File, io::BufReader};

use common::Rand;

use montyformat::{chess::Position, MontyValueFormat};
use tch::{Device, Kind, Tensor};

use crate::arch::{self, ValueNetwork, TOKENS};

pub struct DataLoader {
    file_path: String,
    buffer_size: usize,
    batch_size: usize,
    device: Device,
}

impl common::DataLoader<ValueNetwork> for DataLoader {
    fn new(path: &str, buffer_size_mb: usize, batch_size: usize, device: Device) -> Self {
        Self {
            file_path: path.to_string(),
            buffer_size: buffer_size_mb * 1024 * 1024 / 128,
            batch_size,
            device,
        }
    }

    fn map_batches<F: FnMut(&(Vec<Tensor>, Tensor, usize)) -> bool>(&self, mut f: F) {
        let mut reusable_buffer = Vec::new();

        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let mut preallocs = PreAllocs::new(self.batch_size);

        'dataloading: loop {
            let mut reader = BufReader::new(File::open(self.file_path.as_str()).unwrap());

            while let Ok(game) = MontyValueFormat::deserialise_from(&mut reader, Vec::new()) {
                parse_into_buffer(game, &mut reusable_buffer);

                if shuffle_buffer.len() + reusable_buffer.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&reusable_buffer);
                } else {
                    shuffle(&mut shuffle_buffer);

                    for batch in shuffle_buffer.chunks(self.batch_size) {
                        let (xs, targets) = Self::get_batch_inputs(self.device, batch, &mut preallocs);

                        let should_break = f(&(xs, targets, batch.len()));

                        if should_break {
                            break 'dataloading;
                        }
                    }

                    shuffle_buffer.clear();
                }
            }
        }
    }
}

impl DataLoader {
    pub fn get_batch_inputs(device: Device, batch: &[(Position, f32)], preallocs: &mut PreAllocs) -> (Vec<Tensor>, Tensor) {
        let (xs, targets) = get_tensors(batch, preallocs);
    
        let xs = xs.iter().map(|x| x.to_device(device).to_dense(None, true).transpose(-2, -1)).collect();
        let targets = targets.to_device(device);
    
        (xs, targets)
    }
}

fn shuffle(data: &mut [(Position, f32)]) {
    let mut rng = Rand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

fn parse_into_buffer(game: MontyValueFormat, buffer: &mut Vec<(Position, f32)>) {
    buffer.clear();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        if data.score.abs() < 2000 {
            buffer.push((pos, game.result));
        }

        pos.make(data.best_move, &castling);
    }
}

pub struct PreAllocs {
    feat_indices: Vec<[Vec<i64>; 2]>,
    targets: Vec<f32>,
}

impl PreAllocs {
    pub fn new(batch_size: usize) -> Self {
        let mut preallocs = PreAllocs {
            feat_indices: Vec::with_capacity(TOKENS as usize),
            targets: Vec::with_capacity(batch_size),
        };

        for _ in 0..TOKENS - 1 {
            preallocs.feat_indices.push([
                Vec::with_capacity(batch_size * 8),
                Vec::with_capacity(batch_size * 8),
            ]);
        }

        preallocs.feat_indices.push([
            Vec::with_capacity(batch_size * 32),
            Vec::with_capacity(batch_size * 32),
        ]);

        preallocs
    }
}

fn get_tensors(batch: &[(Position, f32)], preallocs: &mut PreAllocs) -> (Vec<Tensor>, Tensor) {
    let batch_size = batch.len();

    preallocs.targets.clear();

    for p in &mut preallocs.feat_indices {
        p[0].clear();
        p[1].clear();
    }

    for (i, (pos, mut target)) in batch.iter().enumerate() {
        if pos.stm() > 0 {
            target = 1.0 - target;
        }

        preallocs.targets.push(target);

        arch::map_value_features(pos, |piece, feat| {
            preallocs.feat_indices[piece][0].push(i as i64);
            preallocs.feat_indices[piece][1].push(feat as i64);
        });
    }

    let mut xs = Vec::with_capacity(TOKENS as usize);

    let mut push_inputs = |p: &[Vec<i64>; 2], inputs: i64| {
        let total_feats = p[0].len();
        let values = Tensor::from_slice(&vec![1f32; total_feats]);
        let indices = Tensor::from_slice2(&[&p[0], &p[1]]);

        let x = Tensor::sparse_coo_tensor_indices_size(
            &indices,
            &values,
            [batch_size as i64, inputs],
            (Kind::Float, Device::Cpu),
            true,
        );

        xs.push(x);
    };

    for p in preallocs.feat_indices.iter().take(TOKENS as usize - 1) {
        push_inputs(p, arch::INPUTS);
    }

    push_inputs(&preallocs.feat_indices[TOKENS as usize - 1], 768);

    let targets = Tensor::from_slice(&preallocs.targets);

    (xs, targets)
}
