use std::{
    fs::File, io::{BufReader, BufWriter, Cursor}, sync::mpsc, time::Instant
};

use bullet::format::{BulletFormat, ChessBoard};

use montyformat::{FastDeserialise, MontyValueFormat};

#[derive(Default)]
struct Stats {
    positions: usize,
    filtered: usize,
    checks: usize,
    caps: usize,
    scores: usize,
    games: usize,
}

fn main() {
    let mut args = std::env::args();
    args.next();

    let inp_path = args.next().unwrap();
    let out_path = args.next().unwrap();

    let mut reader = BufReader::new(File::open(inp_path).unwrap());
    let mut writer = BufWriter::new(File::create(out_path).unwrap());

    let timer = Instant::now();

    let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(256);

    std::thread::spawn(move || {
        loop {
            let mut buffer = Vec::new();
            if let Ok(()) = MontyValueFormat::deserialise_fast_into_buffer(&mut reader, &mut buffer) {
                sender.send(buffer).unwrap();
            } else {
                break;
            }
        }
    });

    let (sender2, receiver2) = mpsc::sync_channel::<Vec<ChessBoard>>(256);

    let lock = std::thread::spawn(move || {
        let mut stats = Stats::default();

        while let Ok(game_bytes) = receiver.recv() {
            let mut reader = Cursor::new(&game_bytes);
            let game = MontyValueFormat::deserialise_from(&mut reader, Vec::new()).unwrap();

            let mut buf = Vec::new();

            let mut pos = game.startpos;
            let castling = &game.castling;

            for result in game.moves {
                let mut write = true;

                if pos.in_check() {
                    write = false;
                    stats.checks += 1;
                }

                if result.best_move.is_capture() {
                    write = false;
                    stats.caps += 1;
                }

                if result.score == i16::MIN || result.score.abs() > 2000 {
                    write = false;
                    stats.scores += 1;
                }

                if write {
                    buf.push(ChessBoard::from_raw(pos.bbs(), pos.stm(), result.score, game.result).unwrap());
                } else {
                    stats.filtered += 1;
                }

                stats.positions += 1;
                if stats.positions % 4194304 == 0 {
                    let elapsed = timer.elapsed().as_secs_f64();
                    let pps = (stats.positions / 1000) as f64 / elapsed;
                    println!("Processed: {}, Time: {elapsed:.2}s, PPS: {pps:.2}k", stats.positions);
                }

                pos.make(result.best_move, castling);
            }

            stats.games += 1;

            sender2.send(buf).unwrap();
        }

        stats
    });

    while let Ok(buf) = receiver2.recv() {
        ChessBoard::write_to_bin(&mut writer, &buf).unwrap();
    }

    let Stats {
        positions,
        filtered,
        checks,
        caps,
        scores,
        games,
    } = lock.join().unwrap();

    println!("Positions: {positions}");
    println!("Games    : {games}");
    println!("Game Len : {:.2}", positions as f64 / games as f64);
    println!("Filtered : {filtered}");
    println!(" - Checks  : {checks}");
    println!(" - Captures: {caps}");
    println!(" - Scores  : {scores}");
    println!("Remaining: {}", positions - filtered);
}
