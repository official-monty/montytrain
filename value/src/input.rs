use std::{str::FromStr, sync::atomic::{AtomicUsize, Ordering}};

use bullet::{format::BulletFormat, inputs};
use montyformat::chess::{Attacks, Castling, Piece, Position, Side};

use crate::{consts::offsets, threats::map_piece_threat};

const TOTAL_THREATS: usize = 2 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + 768;

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

fn map_features<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let mut bbs = pos.bbs();
    
    // flip to stm perspective
    if pos.stm() == Side::WHITE {
        bbs.swap(0, 1);
        for bb in bbs.iter_mut() {
            *bb = bb.swap_bytes()
        }
    }

    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb= flip_horizontal(*bb);
        }
    };

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

                f(TOTAL_THREATS + [0, 384][side] + 64 * (piece - 2) + sq);
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

#[derive(Clone, Copy)]
pub struct DataPoint {
    pub pos: Position,
    pub result: f32,
    pub score: i16,
}

impl FromStr for DataPoint {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self {
            pos: Position::parse_fen(s, &mut Castling::default()),
            result: 0.5,
            score: 0,
        })
    }
}

impl BulletFormat for DataPoint {
    type FeatureType = usize;

    const HEADER_SIZE: usize = 0;

    fn set_result(&mut self, result: f32) {
        self.result = result;
    }

    fn result(&self) -> f32 {
        if self.pos.stm() == 0 { self.result } else { 1.0 - self.result }
    }

    fn result_idx(&self) -> usize {
        (2.0 * self.result()) as usize
    }

    fn score(&self) -> i16 {
        self.score
    }
}

impl IntoIterator for DataPoint {
    type IntoIter = ThreatInputsIter;
    type Item = (usize, usize);

    fn into_iter(self) -> Self::IntoIter {
        let mut res = ThreatInputsIter {
            features: [0; 128],
            active: 0,
            curr: 0,
        };

        map_features(&self.pos, |feat| {
            res.features[res.active] = feat;
            res.active += 1;
        });

        res
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;

pub struct ThreatInputsIter {
    features: [usize; 128],
    active: usize,
    curr: usize,
}

impl inputs::InputType for ThreatInputs {
    type RequiredDataType = DataPoint;
    type FeatureIter = ThreatInputsIter;

    fn buckets(&self) -> usize {
        1
    }

    fn max_active_inputs(&self) -> usize {
        128
    }

    fn inputs(&self) -> usize {
        TOTAL
    }

    fn feature_iter(&self, pos: &Self::RequiredDataType) -> Self::FeatureIter {
        pos.into_iter()
    }
}

impl Iterator for ThreatInputsIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr;

        if curr == self.active {
            None
        } else {
            self.curr += 1;
            let feat = self.features[curr];
            Some((feat, feat))
        }
    }
}
