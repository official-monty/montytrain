use std::sync::atomic::{AtomicUsize, Ordering};

use bullet::default::{
    formats::{
        bulletformat::ChessBoard,
        montyformat::chess::{Attacks, Piece, Side},
    },
    inputs::{self, Chess768, Factorised, Factorises},
};

use crate::{consts::offsets, threats::map_piece_threat};

pub const NUM_INPUT_BUCKETS: usize = 13;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  8,  9,  9,
    10, 10, 10, 10,
    11, 11, 11, 11,
    11, 11, 11, 11,
    12, 12, 12, 12,
    12, 12, 12, 12,
];

const TOTAL_THREATS: usize = 2 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + 768 * NUM_INPUT_BUCKETS;

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

fn map_features_bucketed<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
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

#[derive(Clone, Copy, Debug)]
pub struct ThreatInputsBucketsMirrored {
    buckets: [usize; 64],
}

impl Default for ThreatInputsBucketsMirrored {
    fn default() -> Self {
        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = BUCKET_LAYOUT[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }
        Self { buckets: expanded }
    }
}

impl inputs::SparseInputType for ThreatInputsBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get = |ksq| {
            let flip = if ksq % 8 > 3 { 7 } else { 0 };
            let bucket = 768 * self.buckets[ksq as usize];
            (flip, bucket)
        };
        let (stm_flip, stm_bucket) = get(pos.our_ksq());
        let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());
        Chess768.map_features(pos, |stm, ntm| {
            f(
                TOTAL_THREATS + stm_bucket + (stm ^ stm_flip),
                TOTAL_THREATS + ntm_bucket + (ntm ^ ntm_flip),
            )
        });

        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features_bucketed(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features_bucketed(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL_THREATS}+{}x{NUM_INPUT_BUCKETS}", Chess768.shorthand())
    }

    fn description(&self) -> String {
        "Threat inputs bucketed mirrored".to_string()
    }
}

impl Factorises<ThreatInputsBucketsMirrored> for Chess768 {
    fn derive_feature(&self, _: &ThreatInputsBucketsMirrored, feat: usize) -> Option<usize> {
        if feat >= TOTAL_THREATS {
            Some((feat - TOTAL_THREATS) % 768)
        } else {
            None
        }
    }
}

type ThreatInputsFactorisedBucketsMirrored = Factorised<ThreatInputsBucketsMirrored, Chess768>;

#[derive(Clone, Copy)]
pub struct ThreatInputs(ThreatInputsFactorisedBucketsMirrored);

impl Default for ThreatInputs {
    fn default() -> Self {
        Self(ThreatInputsFactorisedBucketsMirrored::from_parts(
            ThreatInputsBucketsMirrored::default(),
            Chess768,
        ))
    }
}

impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        self.0.num_inputs()
    }

    fn max_active(&self) -> usize {
        self.0.max_active()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.0.map_features(pos, f)
    }

    fn shorthand(&self) -> String {
        self.0.shorthand()
    }

    fn description(&self) -> String {
        self.0.description()
    }

    fn is_factorised(&self) -> bool {
        self.0.is_factorised()
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        self.0.merge_factoriser(unmerged)
    }
}
