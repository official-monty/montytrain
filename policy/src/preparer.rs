use bullet::default::{formats::montyformat::chess::Move, loader::DataLoader as Blah};

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

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        f: F,
    ) {
        self.loader.map_batches(start_batch, batch_size, f);
    }

    fn prepare(&self, data: &[Self::DataType], threads: usize, _: f32) -> Self::PreparedData {
        PreparedData::new(data, threads)
    }
}

pub struct DenseInput {
    pub value: Vec<f32>,
}

#[derive(Clone)]
pub struct SparseInput {
    pub value: Vec<i32>,
    pub max_active: usize,
}

pub struct PreparedData {
    pub batch_size: usize,
    pub stm: SparseInput,
    pub ntm: SparseInput,
    pub mask: SparseInput,
    pub dist: DenseInput,
}

impl PreparedData {
    pub fn new(data: &[DecompressedData], threads: usize) -> Self {
        let batch_size = data.len();
        let chunk_size = batch_size.div_ceil(threads);

        let mut prep = Self {
            batch_size,
            stm: SparseInput {
                max_active: MAX_ACTIVE,
                value: vec![0; MAX_ACTIVE * batch_size],
            },
            ntm: SparseInput {
                max_active: MAX_ACTIVE,
                value: vec![0; MAX_ACTIVE * batch_size],
            },
            mask: SparseInput {
                max_active: MAX_MOVES,
                value: vec![0; MAX_MOVES * batch_size],
            },
            dist: DenseInput {
                value: vec![0.0; MAX_MOVES * batch_size],
            },
        };

        std::thread::scope(|s| {
            for ((((data_chunk, stm_chunk), ntm_chunk), mask_chunk), dist_chunk) in data
                .chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(MAX_ACTIVE * chunk_size))
                .zip(prep.ntm.value.chunks_mut(MAX_ACTIVE * chunk_size))
                .zip(prep.mask.value.chunks_mut(MAX_MOVES * chunk_size))
                .zip(prep.dist.value.chunks_mut(MAX_MOVES * chunk_size))
            {
                s.spawn(move || {
                    for (i, point) in data_chunk.iter().enumerate() {
                        let input_offset = MAX_ACTIVE * i;
                        let mask_offset = MAX_MOVES * i;
                        let dist_offset = MAX_MOVES * i;

                        let mut j = 0;
                        map_policy_inputs(&point.pos, point.pos.stm(), |feat| {
                            assert!(feat < INPUT_SIZE);
                            stm_chunk[input_offset + j] = feat as i32;
                            j += 1;
                        });

                        for j in j..MAX_ACTIVE {
                            stm_chunk[input_offset + j] = -1;
                        }

                        let j1 = j;

                        let mut j = 0;
                        map_policy_inputs(&point.pos, point.pos.stm() ^ 1, |feat| {
                            assert!(feat < INPUT_SIZE);
                            ntm_chunk[input_offset + j] = feat as i32;
                            j += 1;
                        });

                        for j in j..MAX_ACTIVE {
                            ntm_chunk[input_offset + j] = -1;
                        }

                        assert_eq!(j, j1);

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
