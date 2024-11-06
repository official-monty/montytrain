use montyformat::chess::{Piece, Position, Side};

pub const INPUT_SIZE: usize = 768;
pub const MAX_ACTIVE: usize = 32;
pub const MAX_MOVES: usize = 96;

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let flip = pos.stm() == Side::BLACK;

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        if flip {
            our_bb = our_bb.swap_bytes();
            opp_bb = opp_bb.swap_bytes();
        }

        while our_bb > 0 {
            f(pc + our_bb.trailing_zeros() as usize);
            our_bb &= our_bb - 1;
        }

        while opp_bb > 0 {
            f(384 + pc + opp_bb.trailing_zeros() as usize);
            opp_bb &= opp_bb - 1;
        }
    }
}
