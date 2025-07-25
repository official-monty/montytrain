use montyformat::chess::{Flag, Move, Piece, Position, Side};

pub const MAX_MOVES: usize = 64;
pub const INPUT_SIZE: usize = 768 * 4;
pub const MAX_ACTIVE_BASE: usize = 32;
pub const NUM_MOVES_INDICES: usize = OFFSETS[5][64] + PROMOS + 2 + 8;

pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = hm ^ if pos.stm() == Side::BLACK { 56 } else { 0 };

    let src = usize::from(mov.src() ^ flip);
    let dst = usize::from(mov.to() ^ flip);

    if mov.is_promo() {
        let ffile = src % 8;
        let tfile = dst % 8;
        let promo_id = 2 * ffile + tfile;

        OFFSETS[5][64] + (PROMOS / 4) * (mov.promo_pc() - Piece::KNIGHT) + promo_id
    } else if mov.flag() == Flag::QS || mov.flag() == Flag::KS {
        let is_ks = usize::from(mov.flag() == Flag::KS);
        let is_hm = usize::from(hm == 0);
        OFFSETS[5][64] + PROMOS + (is_ks ^ is_hm)
    } else if mov.flag() == Flag::DBL {
        OFFSETS[5][64] + PROMOS + 2 + (src % 8)
    } else {
        let pc = pos.get_pc(1 << mov.src()) - 2;
        let below = DESTINATIONS[src][pc] & ((1 << dst) - 1);

        OFFSETS[pc][src] + below.count_ones() as usize
    }
}

pub fn map_base_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let vert = if pos.stm() == Side::BLACK { 56 } else { 0 };
    let hori = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let flip = vert ^ hori;

    let threats = pos.threats_by(pos.stm() ^ 1);
    let defences = pos.threats_by(pos.stm());

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        while our_bb > 0 {
            let sq = our_bb.trailing_zeros() as usize;
            let mut feat = pc + (sq ^ flip);

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
            let sq = opp_bb.trailing_zeros() as usize;
            let mut feat = 384 + pc + (sq ^ flip);

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

pub const PROMOS: usize = 4 * 22;

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

const OFFSETS: [[usize; 65]; 6] = {
    let mut offsets = [[0; 65]; 6];

    let mut curr = 0;

    let mut pc = 0;
    while pc < 6 {
        let mut sq = 0;

        while sq < 64 {
            offsets[pc][sq] = curr;
            curr += DESTINATIONS[sq][pc].count_ones() as usize;
            sq += 1;
        }

        offsets[pc][64] = curr;

        pc += 1;
    }

    offsets
};

const DESTINATIONS: [[u64; 6]; 64] = init!(|sq, 64| [PAWN[sq], KNIGHT[sq], bishop(sq), rook(sq), queen(sq), KING[sq]]);

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

const PAWN: [u64; 64] = init!(|sq, 64| {
    let bit = 1 << sq;
    ((bit & !A) << 7) | (bit << 8) | ((bit & !H) << 9)
});

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const fn bishop(sq: usize) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;

    DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
}

const fn rook(sq: usize) -> u64 {
    let rank = sq / 8;
    let file = sq % 8;

    (0xFF << (rank * 8)) ^ (A << file)
}

const fn queen(sq: usize) -> u64 {
    bishop(sq) | rook(sq)
}

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !A) >> 1) | ((k & !H) << 1);
    k ^ (1 << sq)
});
