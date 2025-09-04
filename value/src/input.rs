use std::sync::atomic::{AtomicUsize, Ordering};

use bullet::default::{
    formats::{
        bulletformat::ChessBoard,
        montyformat::chess::{Attacks, Piece, Side},
    },
    inputs,
};

use crate::{consts::offsets, threats::map_piece_threat};

const BUCKET_LAYOUT: [usize; 32] = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 8, 9, 9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    11, 11, 11, 11,
    12, 12, 12, 12,
    12, 12, 12, 12,
];

const TOTAL_THREATS: usize = 2 * offsets::END;
const KING_BUCKETS: usize = 13;
const PST_FACTOR_SIZE: usize = 768;
const TOTAL: usize = TOTAL_THREATS + 768 * KING_BUCKETS + PST_FACTOR_SIZE;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static SQRED: AtomicUsize = AtomicUsize::new(0);
static EVALS: AtomicUsize = AtomicUsize::new(0);
static MAX: AtomicUsize = AtomicUsize::new(0);
const TRACK: bool = false;

pub fn print_feature_stats() {
    let count = COUNT.load(Ordering::Relaxed);
    let sqred = SQRED.load(Ordering::Relaxed);
    let evals = EVALS.load(Ordering::Relaxed);
    let max = MAX.load(Ordering::Relaxed);

    let mean = count as f64 / evals as f64;
    let var = sqred as f64 / evals as f64 - mean.powi(2);
    let pct = 1.96 * var.sqrt();

    println!("Total Evals: {evals}");
    println!("Maximum Active Features: {max}");
    println!("Active Features: {mean:.3} +- {pct:.3} (95%)");
}

fn map_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horizontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros() as usize;
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let wk = (bbs[0] & bbs[Piece::KING]).trailing_zeros() as usize;
    let bk = (bbs[1] & bbs[Piece::KING]).trailing_zeros() as usize;    
    let king_buckets = [
        BUCKET_LAYOUT[4 * (wk >> 3) + (wk & 3)],
        BUCKET_LAYOUT[4 * (bk >> 3) + (bk & 3)],
    ];

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                let bucket = king_buckets[side];
                let base = [0, 384][side] + 64 * (piece - 2) + sq;
                f(TOTAL_THREATS + bucket * 768 + base);
                count += 1;
                f(TOTAL_THREATS + KING_BUCKETS * 768 + base);
                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        map_features(bbs, |stm| f(stm, stm));
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL}")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}
