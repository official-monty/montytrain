use std::{
    fs::File,
    io::BufReader,
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

use montyformat::{
    chess::{Castling, Position}, FastDeserialise, MontyFormat
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
    threads: usize,
}

impl DataReader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self {
            file_path: path.to_string(),
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<DecompressedData>() / 2,
            threads,
        }
    }
}

impl DataReader {
    pub fn map_batches<F: FnMut(&[DecompressedData]) -> bool>(&self, batch_size: usize, mut f: F) {
        let file_path = self.file_path.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let games_per_thread = 1024;

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<u8>>(32);

        std::thread::spawn(move || {
            let mut buffer = Vec::new();

            'dataloading: loop {
                let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                while let Ok(()) = MontyFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                    if game_sender.send(buffer.clone()).is_err() {
                        break 'dataloading;
                    }
                }
            }
        });

        let (mini_sender, mini_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(4);

        std::thread::spawn(move || {
            let mut buffer = Vec::new();

            while let Ok(game_bytes) = game_receiver.recv() {
                buffer.push(game_bytes);
                if buffer.len() == games_per_thread * threads {
                    if std::thread::scope(|s| {
                        let mut handles = Vec::new();

                        for chunk in buffer.chunks(games_per_thread) {
                            let this_sender = mini_sender.clone();
                            let handle = s.spawn(move || {
                                let mut buf = Vec::new();

                                for game_bytes in chunk {
                                    let mut cursor = std::io::Cursor::new(game_bytes);
                                    let game = MontyFormat::deserialise_from(&mut cursor).unwrap();
                                    parse_into_buffer(game, &mut buf);
                                }

                                this_sender.send(buf).is_err()
                            });

                            handles.push(handle);
                        }

                        handles.into_iter().any(|x| x.join().unwrap())
                    }) {
                        break;
                    }

                    buffer.clear();
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<DecompressedData>>(0);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::new();
            shuffle_buffer.reserve_exact(buffer_size);

            while let Ok(buffer) = mini_receiver.recv() {
                if shuffle_buffer.len() + buffer.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&buffer);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    shuffle_buffer.extend_from_slice(&buffer[..diff]);

                    shuffle(&mut shuffle_buffer);

                    if buffer_sender.send(shuffle_buffer).is_err() {
                        break;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                }
            }
        });

        'dataloading: while let Ok(inputs) = buffer_receiver.recv() {
            for batch in inputs.chunks(batch_size) {
                if f(batch) {
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
