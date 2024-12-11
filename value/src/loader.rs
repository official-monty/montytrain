use std::{
    fs::File,
    io::BufReader,
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::input::DataPoint;

use bullet::loader::DataLoader;
use montyformat::MontyValueFormat;

#[derive(Clone)]
pub struct BinpackLoader {
    file_path: [String; 1],
    buffer_size: usize,
}

impl BinpackLoader {
    pub fn new(path: &str, buffer_size_mb: usize) -> Self {
        Self {
            file_path: [path.to_string(); 1],
            buffer_size: buffer_size_mb * 1024 * 1024 / 80 / 2,
        }
    }
}

impl DataLoader<DataPoint> for BinpackLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_path
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[DataPoint]) -> bool>(&self, batch_size: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<DataPoint>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let file_path = self.file_path[0].clone();
        let buffer_size = self.buffer_size;

        std::thread::spawn(move || {
            let mut reusable_buffer = Vec::new();

            'dataloading: loop {
                let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                while let Ok(game) = MontyValueFormat::deserialise_from(&mut reader, Vec::new()) {
                    if buffer_msg_receiver.try_recv().unwrap_or(false) {
                        break 'dataloading;
                    }

                    parse_into_buffer(game, &mut reusable_buffer);

                    if shuffle_buffer.len() + reusable_buffer.len() < shuffle_buffer.capacity() {
                        shuffle_buffer.extend_from_slice(&reusable_buffer);
                    } else {
                        let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                        if diff > 0 {
                            shuffle_buffer.extend_from_slice(&reusable_buffer[..diff]);
                        }

                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::new();
                        shuffle_buffer.reserve_exact(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(inputs) = buffer_receiver.recv() {
            for batch in inputs.chunks(batch_size) {
                let should_break = f(batch);

                if should_break {
                    buffer_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        }

        drop(buffer_receiver);
    }
}

fn parse_into_buffer(game: MontyValueFormat, buffer: &mut Vec<DataPoint>) {
    buffer.clear();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        if data.score.abs() < 2000 && data.score != i16::MIN {
            buffer.push(
                DataPoint {
                    pos,
                    result: game.result,
                    score: data.score,
                }
            );
        }

        pos.make(data.best_move, &castling);
    }
}

fn shuffle(data: &mut [DataPoint]) {
    let mut rng = Rand::with_seed();

    for i in (0..data.len()).rev() {
        let idx = rng.rng() as usize % (i + 1);
        data.swap(idx, i);
    }
}

pub struct Rand(u64);

impl Rand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u64
            & 0xFFFF_FFFF;

        Self(seed)
    }

    pub fn rng(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}
