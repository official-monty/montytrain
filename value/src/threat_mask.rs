use bullet::montyformat::chess::{Attacks, Piece, Position, Side};

pub fn map_threat_mask<F: FnMut(usize)>(pos: &Position, ntm: bool, mut f: F) {
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

pub const MAX_THREATS: usize = 128;
pub const NUM_THREATS: usize = OFFSETS[64];

pub fn map_threat_to_index(src: usize, dst: usize) -> usize {
    let below = ALL_DESTINATIONS[src] & ((1 << dst) - 1);
    OFFSETS[src] + below.count_ones() as usize
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