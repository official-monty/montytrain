use std::{
    fs::File,
    io::{BufReader, Cursor},
    sync::mpsc::{self, SyncSender},
    time::{SystemTime, UNIX_EPOCH},
};

use bullet::{format::ChessBoard, loader::DataLoader};
use montyformat::{FastDeserialise, MontyValueFormat};

#[derive(Clone)]
pub struct BinpackLoader {
    file_path: [String; 1],
    buffer_size: usize,
    threads: usize,
}

impl BinpackLoader {
    pub fn new(path: &str, buffer_size_mb: usize, threads: usize) -> Self {
        Self {
            file_path: [path.to_string(); 1],
            buffer_size: buffer_size_mb * 1024 * 1024 / 32 / 2,
            threads,
        }
    }
}

impl DataLoader<ChessBoard> for BinpackLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_path
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[ChessBoard]) -> bool>(&self, batch_size: usize, mut f: F) {
        let mut shuffle_buffer = Vec::new();
        shuffle_buffer.reserve_exact(self.buffer_size);

        let file_path = self.file_path[0].clone();
        let buffer_size = self.buffer_size;

        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(256);
        let (msg_sender, msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: loop {
                let mut reader = BufReader::new(File::open(file_path.as_str()).unwrap());

                let mut buffer = Vec::new();
                while let Ok(()) = MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                    if msg_receiver.try_recv().unwrap_or(false) || sender.send(buffer).is_err() {
                        break 'dataloading;
                    }

                    buffer = Vec::new();
                }
            }
        });

        let (game_sender, game_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * self.threads);
        let (game_msg_sender, game_msg_receiver) = mpsc::sync_channel::<bool>(1);

        let threads = self.threads;

        std::thread::spawn(move || {
            let mut reusable = Vec::new();
            'dataloading: while let Ok(game_bytes) = receiver.recv() {
                if game_msg_receiver.try_recv().unwrap_or(false) {
                    msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                reusable.push(game_bytes);

                if reusable.len() % (8192 * threads) == 0 {
                    convert_buffer(threads, &game_sender, &reusable);
                    reusable.clear();
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            'dataloading: while let Ok(game) = game_receiver.recv() {
                if buffer_msg_receiver.try_recv().unwrap_or(false) {
                    game_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                if shuffle_buffer.len() + game.len() < shuffle_buffer.capacity() {
                    shuffle_buffer.extend_from_slice(&game);
                } else {
                    let diff = shuffle_buffer.capacity() - shuffle_buffer.len();
                    if diff > 0 {
                        shuffle_buffer.extend_from_slice(&game[..diff]);
                    }

                    shuffle(&mut shuffle_buffer);

                    if buffer_msg_receiver.try_recv().unwrap_or(false)
                        || buffer_sender.send(shuffle_buffer).is_err()
                    {
                        game_msg_sender.send(true).unwrap();
                        break 'dataloading;
                    }

                    shuffle_buffer = Vec::new();
                    shuffle_buffer.reserve_exact(buffer_size);
                    shuffle_buffer.extend_from_slice(&game[diff..]);
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {            
            for batch in shuffle_buffer.chunks(batch_size) {
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

fn convert_buffer(threads: usize, sender: &SyncSender<Vec<ChessBoard>>, games: &[Vec<u8>]) {
    let chunk_size = games.len().div_ceil(threads);

    std::thread::scope(|s| {
        for chunk in games.chunks(chunk_size) {
            let this_sender = sender.clone();
            s.spawn(move || {
                let mut buffer = Vec::new();

                for game_bytes in chunk {
                    parse_into_buffer(game_bytes, &mut buffer);
                }

                this_sender.send(buffer)
            });
        }
    });
}

fn parse_into_buffer(game_bytes: &[u8], buffer: &mut Vec<ChessBoard>) {
    let mut reader = Cursor::new(game_bytes);
    let game = MontyValueFormat::deserialise_from(&mut reader, Vec::new()).unwrap();

    let mut pos = game.startpos;
    let castling = game.castling;

    for data in game.moves {
        buffer.push(ChessBoard::from_raw(pos.bbs(), pos.stm(), data.score, game.result).unwrap());

        pos.make(data.best_move, &castling);
    }
}

fn shuffle(data: &mut [ChessBoard]) {
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
