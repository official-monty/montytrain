use montyformat::chess::{Move, Position, Side};

pub const MAX_MOVES: usize = 96;
pub const NUM_MOVES: usize = OFFSETS[64];

//pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
//    let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
//    let pc = pos.get_pc(1 << mov.src()) - Piece::PAWN;
//    let dest = usize::from(mov.to() ^ flip);
//
//    64 * pc + dest
//}

#[allow(unused)]
pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let from = usize::from(mov.src() ^ flip);
    let dest = usize::from(mov.to() ^ flip);

    let below = ALL_DESTINATIONS[from] & ((1 << dest) - 1);

    OFFSETS[from] + below.count_ones() as usize
}

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

const OFFSETS: [usize; 65] = {
    let mut offsets = [0; 65];

    let mut curr = 0;
    let mut sq = 0;

    while sq < 64 {
        offsets[sq] = curr;
        curr += ALL_DESTINATIONS[sq].count_ones() as usize;
        sq += 1;
    }

    offsets[64] = curr;

    offsets
};

const ALL_DESTINATIONS: [u64; 64] = init!(|sq, 64| {
    let rank = sq / 8;
    let file = sq % 8;

    let rooks = (0xFF << (rank * 8)) ^ (A << file);
    let bishops = DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank];

    rooks | bishops | KNIGHT[sq] | KING[sq]
});

const A: u64 = 0x0101_0101_0101_0101;
const H: u64 = A << 7;


const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !A) >> 1) | ((k & !H) << 1);
    k ^ (1 << sq)
});

#[cfg(test)]
mod test {
    use montyformat::chess::{Flag, Move, Position};

    use crate::moves::{map_move_to_index, NUM_MOVES};

    #[test]
    fn test_count() {
        let mov = Move::new(63, 62, Flag::QUIET);
        let pos = Position::default();
        assert_eq!(map_move_to_index(&pos, mov), NUM_MOVES - 1);
    }
}
