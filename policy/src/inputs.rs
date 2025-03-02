use bullet::default::formats::montyformat::chess::{Piece, Position, Side};

pub const INPUT_SIZE: usize = 768;
pub const MAX_ACTIVE: usize = 32;

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Position, side: usize, mut f: F) {
    let flip = side == Side::BLACK;
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(side);
        let mut opp_bb = pos.piece(piece) & pos.piece(side ^ 1);

        if flip {
            our_bb = our_bb.swap_bytes();
            opp_bb = opp_bb.swap_bytes();
        }

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
