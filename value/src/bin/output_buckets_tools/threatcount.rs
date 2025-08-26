use std::{
    fs::File,
    io::{BufWriter, Write},
};

use bullet::{
    default::{inputs::SparseInputType, loader::{self, DataLoader}},
    game::formats::bulletformat::ChessBoard,
};

#[path = "../consts.rs"]
mod consts;
#[path = "../input.rs"]
mod input;
#[path = "../threats.rs"]
mod threats;

use consts::offsets;
use input::ThreatInputs;

const TOTAL_THREATS: usize = 2 * offsets::END;
const MAX_POSITIONS: u64 = 100_000_000;

fn main() {
    let mut args = std::env::args();
    args.next();

    let bin_path = args.next().expect("binpack path");
    let log_path = args.next().expect("log output path");

    let mut writer = BufWriter::new(File::create(log_path).expect("create log"));

    let loader = loader::MontyBinpackLoader::new(&bin_path, 96000, 8, |_p, _m, _s, _r| true);
    let inputs = ThreatInputs;
    let mut seen = 0u64;

    loader.map_batches(0, 4096, |batch: &[ChessBoard]| {
        for board in batch {
            if seen >= MAX_POSITIONS {
                return true;
            }

            let mut pieces = 0usize;
            let mut threats = 0usize;

            inputs.map_features(board, |idx, _| {
                if idx < TOTAL_THREATS {
                    threats += 1;
                } else {
                    pieces += 1;
                }
            });

            writeln!(writer, "{}, {}", pieces, threats).expect("write log");
            seen += 1;
        }
        false
    });
}
