mod consts;
mod input;
mod threats;

use consts::indices;
use input::ThreatInputs;

use bullet::trainer::default::inputs::SparseInputType;

fn main() {
    println!("Attacks:");
    println!("Pawn   : {}", indices::PAWN);
    println!("Bishop : {}", indices::BISHOP[64]);
    println!("Knight : {}", indices::KNIGHT[64]);
    println!("Rook   : {}", indices::ROOK[64]);
    println!("Queen  : {}", indices::QUEEN[64]);
    println!("King   : {}", indices::KING[64]);

    println!("Inputs: {}", ThreatInputs.num_inputs());

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        //"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        //"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        //"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        println!();
        println!("{fen}");
        ThreatInputs.map_features(&pos, |stm, ntm| {
            print!("({stm} {ntm}) ");
        });
        println!();
    }
}
