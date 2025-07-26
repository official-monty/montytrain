use bullet_core::{
    graph::builder::Shape,
    trainer::{
        dataloader::{DataLoader, HostDenseMatrix, HostMatrix, HostSparseMatrix, PreparedBatchHost},
        DataLoadingError,
    },
};
use montyformat::chess::Move;

use super::reader::{DataReader, DecompressedData};
use crate::{
    inputs::{self, INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES, NUM_MOVES_INDICES},
    see,
};

#[derive(Clone)]
pub struct MontyDataLoader {
    reader: DataReader,
    threads: usize,
}

impl MontyDataLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self { reader: DataReader::new(path, buffer_size_mb), threads }
    }
}

impl DataLoader for MontyDataLoader {
    type Error = DataLoadingError;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(self, batch_size: usize, mut f: F) -> Result<(), Self::Error> {
        let mut pool = rayon::ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap();

        self.reader.map_batches(batch_size, |batch| f(prepare(batch, self.threads, &mut pool)));

        Ok(())
    }
}

pub fn prepare(data: &[DecompressedData], threads: usize, pool: &mut rayon::ThreadPool) -> PreparedBatchHost {
    let batch_size = data.len();
    let chunk_size = batch_size.div_ceil(threads);

    let t = std::time::Instant::now();

    let mut inputs = vec![0; MAX_ACTIVE_BASE * batch_size];
    let mut see_inputs = vec![0; MAX_ACTIVE_BASE * 64 * batch_size];
    let mut stms = vec![0; MAX_MOVES * batch_size];
    let mut moves = vec![0; MAX_MOVES * batch_size];
    let mut dist = vec![0.0; MAX_MOVES * batch_size];

    pool.scope(|s| {
        for (((((data_chunk, input_chunk), see_input_chunk), stms_chunk), moves_chunk), dist_chunk) in data
            .chunks(chunk_size)
            .zip(inputs.chunks_mut(MAX_ACTIVE_BASE * chunk_size))
            .zip(see_inputs.chunks_mut(MAX_ACTIVE_BASE * 64 * chunk_size))
            .zip(stms.chunks_mut(MAX_MOVES * chunk_size))
            .zip(moves.chunks_mut(MAX_MOVES * chunk_size))
            .zip(dist.chunks_mut(MAX_MOVES * chunk_size))
        {
            s.spawn(move |_| {
                for (i, point) in data_chunk.iter().enumerate() {
                    let input_offset = MAX_ACTIVE_BASE * i;
                    let moves_offset = MAX_MOVES * i;
                    let see_input_offset = MAX_ACTIVE_BASE * 64 * i;

                    let mut j = 0;
                    inputs::map_base_inputs(&point.pos, None, |feat| {
                        assert!(feat < INPUT_SIZE);
                        input_chunk[input_offset + j] = feat as i32;
                        j += 1;
                    });

                    for k in j..MAX_ACTIVE_BASE {
                        input_chunk[input_offset + k] = -1;
                    }

                    assert!(j <= MAX_ACTIVE_BASE, "More inputs provided than the specified maximum!");

                    let mut total = 0;
                    let mut distinct = 0;

                    let pos = &point.pos;

                    let stm = point.pos.stm();
                    let hm = if point.pos.king_index() % 8 > 3 { 7 } else { 0 };
                    let no_threats = Some((stm, hm));

                    for &(mov, visits) in &point.moves[..point.num] {
                        total += visits;

                        let mov = Move::from(mov);
                        moves_chunk[moves_offset + distinct] = inputs::map_move_to_index(pos, mov) as i32;
                        dist_chunk[moves_offset + distinct] = f32::from(visits);

                        let resolved = see::get_resolved_see_pos(pos, &point.castling, mov);
                        stms_chunk[moves_offset + distinct] = i32::from(resolved.stm() == stm);

                        let mut k = 0;
                        inputs::map_base_inputs(&resolved, no_threats, |feat| {
                            assert!(feat < 768);
                            see_input_chunk[see_input_offset + MAX_ACTIVE_BASE * distinct + k] = feat as i32;
                            k += 1;
                        });

                        for l in k..MAX_ACTIVE_BASE {
                            see_input_chunk[see_input_offset + MAX_ACTIVE_BASE * distinct + l] = -1;
                        }

                        distinct += 1;
                    }

                    for k in distinct..MAX_MOVES {
                        stms_chunk[moves_offset + k] = -1;
                        moves_chunk[moves_offset + k] = -1;

                        for l in 0..MAX_ACTIVE_BASE {
                            see_input_chunk[see_input_offset + MAX_ACTIVE_BASE * k + l] = -1;
                        }
                    }

                    let total = f32::from(total);

                    for idx in 0..distinct {
                        dist_chunk[moves_offset + idx] /= total;
                    }
                }
            });
        }
    });

    let mut prep = PreparedBatchHost { batch_size, inputs: Default::default() };

    unsafe {
        prep.inputs.insert(
            "inputs".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(inputs, batch_size, Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE)),
        );

        prep.inputs.insert(
            "see_inputs".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(
                see_inputs,
                batch_size,
                Shape::new(768, MAX_MOVES),
                MAX_MOVES * MAX_ACTIVE_BASE,
            )),
        );

        prep.inputs.insert(
            "stms".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(stms, batch_size, Shape::new(MAX_MOVES, 1), MAX_MOVES)),
        );

        prep.inputs.insert(
            "moves".to_string(),
            HostMatrix::Sparse(HostSparseMatrix::new(moves, batch_size, Shape::new(NUM_MOVES_INDICES, 1), MAX_MOVES)),
        );
    }

    prep.inputs.insert(
        "targets".to_string(),
        HostMatrix::Dense(HostDenseMatrix::new(dist, batch_size, Shape::new(MAX_MOVES, 1))),
    );

    println!("{}", batch_size as f64 / t.elapsed().as_secs_f64());

    prep
}
