use bullet::{
    default::loader::{DataLoader, MontyBinpackLoader},
    format::{BulletFormat, ChessBoard},
    montyformat::chess::{Move, Position},
    Shape,
};

use crate::{
    inputs::{self, INPUT_SIZE, MAX_ACTIVE},
    threats::{MAX_THREATS, NUM_THREATS},
};

#[derive(Clone)]
pub struct DataPreparer<T: Fn(&Position, Move, i16, f32) -> bool> {
    loader: MontyBinpackLoader<T>,
}

impl<T: Fn(&Position, Move, i16, f32) -> bool> DataPreparer<T> {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize, f: T) -> Self {
        Self { loader: MontyBinpackLoader::new(path, buffer_size_mb, threads, f) }
    }
}

impl<T> bullet::DataPreparer for DataPreparer<T>
where
    T: Fn(&Position, Move, i16, f32) -> bool + Clone + Send + Sync + 'static,
{
    type DataType = ChessBoard;
    type PreparedData = PreparedData;

    fn get_data_file_paths(&self) -> &[String] {
        self.loader.data_file_paths()
    }

    fn try_count_positions(&self) -> Option<u64> {
        self.loader.count_positions()
    }

    fn load_and_map_batches<F: FnMut(&[Self::DataType]) -> bool>(&self, start_batch: usize, batch_size: usize, f: F) {
        self.loader.map_batches(start_batch, batch_size, f);
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
    pub stm_mask: SparseInput,
    pub ntm_mask: SparseInput,
    pub target: DenseInput,
}

impl PreparedData {
    pub fn new(data: &[ChessBoard], threads: usize) -> Self {
        let batch_size = data.len();
        let chunk_size = batch_size.div_ceil(threads);

        let mut prep = Self {
            batch_size,
            inputs: SparseInput {
                shape: Shape::new(INPUT_SIZE, batch_size),
                max_active: MAX_ACTIVE,
                value: vec![0; MAX_ACTIVE * batch_size],
            },
            stm_mask: SparseInput {
                shape: Shape::new(NUM_THREATS, batch_size),
                max_active: MAX_THREATS,
                value: vec![0; MAX_THREATS * batch_size],
            },
            ntm_mask: SparseInput {
                shape: Shape::new(NUM_THREATS, batch_size),
                max_active: MAX_THREATS,
                value: vec![0; MAX_THREATS * batch_size],
            },
            target: DenseInput { shape: Shape::new(3, batch_size), value: vec![0.0; 3 * batch_size] },
        };

        std::thread::scope(|s| {
            for ((((data_chunk, input_chunk), stm_mask_chunk), ntm_mask_chunk), target_chunk) in data
                .chunks(chunk_size)
                .zip(prep.inputs.value.chunks_mut(MAX_ACTIVE * chunk_size))
                .zip(prep.stm_mask.value.chunks_mut(MAX_THREATS * chunk_size))
                .zip(prep.ntm_mask.value.chunks_mut(MAX_THREATS * chunk_size))
                .zip(prep.target.value.chunks_mut(3 * chunk_size))
            {
                s.spawn(move || {
                    for (i, point) in data_chunk.iter().enumerate() {
                        let input_offset = MAX_ACTIVE * i;
                        let mask_offset = MAX_THREATS * i;
                        let target_offset = 3 * i;

                        let mut bb = [0; 8];
                        for (pc, sq) in point.into_iter() {
                            let bit = 1 << sq;
                            bb[usize::from(pc >> 3) & 1] |= bit;
                            bb[2 + usize::from(pc & 7)] |= bit;
                        }

                        let pos = Position::from_raw(bb, false, 0, 0, 0, 1);

                        let mut j = 0;
                        inputs::map_inputs(&pos, |feat| {
                            assert!(feat < INPUT_SIZE);
                            input_chunk[input_offset + j] = feat as i32;
                            j += 1;
                        });

                        if j < MAX_ACTIVE {
                            input_chunk[input_offset + j] = -1;
                        }

                        j = 0;
                        inputs::map_threats(&pos, false, |feat| {
                            assert!(feat < NUM_THREATS);
                            stm_mask_chunk[mask_offset + j] = feat as i32;
                            j += 1;
                        });

                        if j < MAX_THREATS {
                            stm_mask_chunk[mask_offset + j] = -1;
                        }

                        j = 0;
                        inputs::map_threats(&pos, true, |feat| {
                            assert!(feat < NUM_THREATS);
                            ntm_mask_chunk[mask_offset + j] = feat as i32;
                            j += 1;
                        });

                        if j < MAX_THREATS {
                            ntm_mask_chunk[mask_offset + j] = -1;
                        }

                        target_chunk[target_offset + point.result_idx()] = 1.0;
                    }
                });
            }
        });

        prep
    }
}
