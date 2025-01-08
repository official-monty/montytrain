use bullet::montyformat::chess::{Attacks, Piece, Position, Side};

use crate::threats::map_threat_to_index;

pub const INPUT_SIZE: usize = 768;
pub const MAX_ACTIVE: usize = 32;

pub fn map_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    assert_eq!(pos.stm(), Side::WHITE);
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        while our_bb > 0 {
            let sq = our_bb.trailing_zeros();
            f(pc + (sq ^ hm) as usize);
            our_bb &= our_bb - 1;
        }

        while opp_bb > 0 {
            let sq = opp_bb.trailing_zeros();
            f(384 + pc + (sq ^ hm) as usize);
            opp_bb &= opp_bb - 1;
        }
    }
}

pub fn map_threats<F: FnMut(usize)>(pos: &Position, ntm: bool, mut f: F) {
    assert_eq!(pos.stm(), Side::WHITE);
    let side = usize::from(ntm);
    let occ = pos.boys() ^ pos.opps();
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };

    let get_feat = |src: usize, dst: usize| map_threat_to_index(src ^ hm, dst ^ hm);

    for piece in Piece::PAWN..=Piece::KING {
        let mut bb = pos.piece(piece) & pos.piece(side);
        while bb > 0 {
            let src = bb.trailing_zeros() as usize;
            bb &= bb - 1;

            let mut threats = match piece {
                Piece::PAWN => Attacks::pawn(src, side),
                Piece::KNIGHT => Attacks::knight(src),
                Piece::BISHOP => Attacks::bishop(src, occ),
                Piece::ROOK => Attacks::rook(src, occ),
                Piece::QUEEN => Attacks::queen(src, occ),
                Piece::KING => Attacks::king(src),
                _ => unreachable!(),
            } & occ;

            while threats > 0 {
                let dst = threats.trailing_zeros() as usize;
                threats &= threats - 1;

                f(get_feat(src, dst));
            }
        }
    }
}
