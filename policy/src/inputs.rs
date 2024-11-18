use montyformat::chess::{Piece, Position, Side};

pub const INPUT_SIZE: usize = 768 * 4;
pub const MAX_ACTIVE: usize = 32;

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let flip = pos.stm() == Side::BLACK;
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };

    let mut threats = pos.threats_by(pos.stm() ^ 1);
    let mut defences = pos.threats_by(pos.stm());

    if flip {
        threats = threats.swap_bytes();
        defences = defences.swap_bytes();
    }

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        if flip {
            our_bb = our_bb.swap_bytes();
            opp_bb = opp_bb.swap_bytes();
        }

        while our_bb > 0 {
            let sq = our_bb.trailing_zeros();
            let mut feat = pc + (sq ^ hm) as usize;

            let bit = 1 << sq;
            if threats & bit > 0 {
                feat += 768;
            }

            if defences & bit > 0 {
                feat += 768 * 2;
            }

            f(feat);

            our_bb &= our_bb - 1;
        }

        while opp_bb > 0 {
            let sq = opp_bb.trailing_zeros();
            let mut feat = 384 + pc + (sq ^ hm) as usize;

            let bit = 1 << sq;
            if threats & bit > 0 {
                feat += 768;
            }

            if defences & bit > 0 {
                feat += 768 * 2;
            }

            f(feat);

            opp_bb &= opp_bb - 1;
        }
    }
}
