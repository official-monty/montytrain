use std::{
    fs::File,
    io::BufReader,
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

use montyformat::{
    chess::{Castling, Position},
    MontyFormat,
};

use crate::inputs::MAX_MOVES;

#[derive(Clone, Copy)]
pub struct DecompressedData {
    pub pos: Position,
    pub castling: Castling,
    pub moves: [(u16, u16); MAX_MOVES],
    pub num: usize,
}

#[derive(Clone)]
pub struct DataReader {
    file_path: String,
    buffer_size: usize,
}

impl DataReader {
    pub fn new(path: &str, buffer_size_mb: usize) -> Self {
        Self {
            file_path: path.to_string(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<DecompressedData>() / 2,
        }
    }
}

impl DataReader {
    pub fn map_batches<F: FnMut(&[DecompressedData]) -> bool>(&self, batch_size: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let file_path = self.file_path.clone();
        let buffer_size = self.buffer_size;

        std::thread::spawn(move || {
            let mut reusable_buffer = Vec::new();

            'dataloading: loop {
                let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                while let Ok(game) = MontyFormat::deserialise_from(&mut reader) {
                    if buffer_msg_receiver.try_recv().unwrap_or(false) {
                        break 'dataloading;
                    }

                    parse_into_buffer(game, &mut reusable_buffer);

                    if shuffle_buffer.len() + reusable_buffer.len() < shuffle_buffer.capacity() {
                        shuffle_buffer.extend_from_slice(&reusable_buffer);
                    } else {
                        let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                        shuffle_buffer.extend_from_slice(&reusable_buffer[..diff]);

                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false) {
                            break 'dataloading;
                        }

                        if buffer_sender.send(shuffle_buffer).is_err() {
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
        if let Some(dist) = data.visit_distribution.as_ref() {
            if dist.len() > 1 && dist.len() <= MAX_MOVES {
                let mut policy_data = DecompressedData { pos, castling, moves: [(0, 0); MAX_MOVES], num: dist.len() };

                for (i, (mov, visits)) in dist.iter().enumerate() {
                    policy_data.moves[i] = (u16::from(*mov), *visits as u16);
                }

                buffer.push(policy_data);
            }
        }

        pos.make(data.best_move, &castling);
    }
}

pub struct Rand(u64);

impl Rand {
    pub fn with_seed() -> Self {
        let seed = SystemTime::now().duration_since(UNIX_EPOCH).expect("Guaranteed increasing.").as_micros() as u64
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
