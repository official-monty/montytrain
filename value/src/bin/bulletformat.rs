use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

use bullet::format::{BulletFormat, ChessBoard};

use montyformat::MontyValueFormat;

fn main() {
    let mut args = std::env::args();
    args.next();

    let inp_path = args.next().unwrap();
    let out_path = args.next().unwrap();

    let mut reader = BufReader::new(File::open(inp_path).unwrap());
    let mut writer = BufWriter::new(File::create(out_path).unwrap());

    let mut buf = Vec::new();

    let mut positions = 0;
    let mut filtered = 0;
    let checks = 0;
    let caps = 0;
    let mut scores = 0;
    let res = 0;
    let mut games = 0;

    loop {
        if let Ok(game) = MontyValueFormat::deserialise_from(&mut reader, Vec::new()) {
            let mut pos = game.startpos;
            let castling = &game.castling;

            for result in game.moves {
                let mut write = true;

                //if board.in_check() {
                //    write = false;
                //    checks += 1;
                //}
    
                //if mov.is_capture() {
                //    write = false;
                //    caps += 1;
                //}
    
                if result.score == i16::MIN || result.score.abs() > 2000 {
                    write = false;
                    scores += 1;
                }
    
                //if write {
                //    let wdl = 1.0 / (1.0 + (-f32::from(score) / 400.0).exp());
                //    if (result - wdl).abs() > 0.6 {
                //        write = false;
                //        res += 1;
                //    }
                //}
    
                if write {
                    buf.push(ChessBoard::from_raw(pos.bbs(), pos.stm(), result.score, game.result).unwrap());
                } else {
                    filtered += 1;
                }
    
                positions += 1;
                if positions % 4194304 == 0 {
                    println!("Processed: {positions}");
                }

                pos.make(result.best_move, castling);
            }
        } else {
            break;
        }

        games += 1;

        ChessBoard::write_to_bin(&mut writer, &buf).unwrap();
        buf.clear();
    }

    println!("Positions: {positions}");
    println!("Games    : {games}");
    println!("Game Len : {:.2}", positions as f64 / games as f64);
    println!("Filtered : {filtered}");
    println!(" - Checks  : {checks}");
    println!(" - Captures: {caps}");
    println!(" - Scores  : {scores}");
    println!(" - Results : {res}");
    println!("Remaining: {}", positions - filtered);
}
