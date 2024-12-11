use std::str::FromStr;

use bullet::{format::BulletFormat, inputs};
use montyformat::chess::{Attacks, Castling, Piece, Position, Side};

use crate::{consts::offsets, threats::map_piece_threat};

const TOTAL_THREATS: usize = 2 * 12 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + 768;

fn map_features<F: FnMut(usize)>(position: &Position, mut f: F) {
    let pos = if position.stm() == Side::WHITE {
        *position
    } else {
        let mut bbs = [0; 8];
        for (new, old) in bbs.iter_mut().zip(position.bbs().iter()) {
            *new = old.swap_bytes()
        }

        bbs.swap(0, 1);

        Position::from_raw(bbs, false, 0, 0, position.halfm(), position.fullm())
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece;
            map_bb(pos.piece(side) & pos.piece(piece), |sq| pieces[sq] = pc);
        }
    }

    let occ = pos.occ();

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = 12 * offsets::END * side;

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(pos.piece(side) & pos.piece(piece), |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                if threats == 0 {
                    f(TOTAL_THREATS + [0, 384][side] + 64 * (piece - 2) + sq)
                } else {
                    map_bb(threats, |dest| {
                        let idx = map_piece_threat(piece, sq, dest);
                        f(side_offset + pieces[dest] * offsets::END + idx)
                    });
                }
            });
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
