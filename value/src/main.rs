mod consts;
mod input;
mod threats;

use input::ThreatInputs;

use bullet::trainer::default::{inputs::SparseInputType, formats::bulletformat::ChessBoard};

fn main() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 | 0 | 0.0";
    let board: ChessBoard = fen.parse().unwrap();

    ThreatInputs.map_features(&board, |stm, ntm| {
        println!("{stm: <5} : {ntm: <5}")
    });
}
