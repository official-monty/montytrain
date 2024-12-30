use bullet::{loader::DataLoader as Blah, Shape};
use montyformat::chess::Move;

use crate::{
    inputs::{map_policy_inputs, INPUT_SIZE, MAX_ACTIVE},
    loader::{DataLoader, DecompressedData},
    moves::{map_move_to_index, MAX_MOVES, NUM_MOVES},
};

#[derive(Clone)]
pub struct DataPreparer {
    loader: DataLoader,
}

impl DataPreparer {
    pub fn new(path: &str, buffer_size_mb: usize) -> Self {
        Self {
            loader: DataLoader::new(path, buffer_size_mb),
        }
    }
}

impl bullet::DataPreparer for DataPreparer {
    type DataType = DecompressedData;
    type PreparedData = PreparedData;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, batch_size: usize, f: F) {
        self.loader.map_batches(batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, _: f32) -> Self::PreparedData {
        PreparedData::new(data, threads)
    }
}

pub struct DenseInput {
    pub shape: Shape,
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseInput {
    pub shape: Shape,
    pub value: Vec<i32>,
    pub max_active: usize,
}

pub struct PreparedData {
    pub batch_size: usize,
    pub inputs: SparseInput,
    pub mask: SparseInput,
    pub dist: DenseInput,
}

impl PreparedData {
    pub fn new(data: &[DecompressedData], threads: usize) -> Self {
        let batch_size = data.len();
        let chunk_size = batch_size.div_ceil(threads);

        let mut prep = Self {
            batch_size,
            inputs: SparseInput {
                shape: Shape::new(INPUT_SIZE, batch_size),
                max_active: MAX_ACTIVE,
                value: vec![0; MAX_ACTIVE * batch_size],
            },
            mask: SparseInput {
                shape: Shape::new(NUM_MOVES, batch_size),
                max_active: MAX_MOVES,
                value: vec![0; MAX_MOVES * batch_size],
            },
            dist: DenseInput {
                shape: Shape::new(MAX_MOVES, batch_size),
                value: vec![0.0; MAX_MOVES * batch_size],
            },
        };

        std::thread::scope(|s| {
            for (((data_chunk, input_chunk), mask_chunk), dist_chunk) in data
                .chunks(chunk_size)
                .zip(prep.inputs.value.chunks_mut(MAX_ACTIVE * chunk_size))
                .zip(prep.mask.value.chunks_mut(MAX_MOVES * chunk_size))
                .zip(prep.dist.value.chunks_mut(MAX_MOVES * chunk_size))
            {
                s.spawn(move || {
                    for (i, point) in data_chunk.iter().enumerate() {
                        let input_offset = MAX_ACTIVE * i;
                        let mask_offset = MAX_MOVES * i;
                        let dist_offset = MAX_MOVES * i;

                        let mut j = 0;
                        map_policy_inputs(&point.pos, |feat| {
                            assert!(feat < INPUT_SIZE);
                            input_chunk[input_offset + j] = feat as i32;
                            j += 1;
                        });

                        if j < MAX_ACTIVE {
                            input_chunk[input_offset + j] = -1;
                        }

                        assert!(
                            j <= MAX_ACTIVE,
                            "More inputs provided than the specified maximum!"
                        );

                        let mut total = 0;
                        let mut distinct = 0;

                        for &(mov, visits) in &point.moves[..point.num] {
                            let idx = map_move_to_index(&point.pos, Move::from(mov));
                            assert!(idx < NUM_MOVES, "{idx} >= {NUM_MOVES}");
                            total += visits;

                            mask_chunk[mask_offset + distinct] = idx as i32;
                            dist_chunk[dist_offset + distinct] = f32::from(visits);
                            distinct += 1;
                        }

                        if distinct < MAX_MOVES {
                            mask_chunk[mask_offset + distinct] = -1;
                        }

                        let total = f32::from(total);

                        for idx in 0..distinct {
                            dist_chunk[dist_offset + idx] /= total;
                        }
                    }
                });
            }
        });

        prep
    }
}
