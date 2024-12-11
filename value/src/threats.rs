use montyformat::chess::Piece;

use crate::consts::{attacks, offsets};

pub fn map_piece_threat(piece: usize, src: usize, dest: usize) -> usize {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest),
        Piece::KNIGHT => map_knight_threat(src, dest),
        Piece::BISHOP => map_bishop_threat(src, dest),
        Piece::ROOK => map_rook_threat(src, dest),
        Piece::QUEEN => map_queen_threat(src, dest),
        Piece::KING => map_king_threat(src, dest),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_pawn_threat(src: usize, dest: usize) -> usize {
    let diff = if dest > src { dest - src } else { src - dest };
    let attack = if diff == 7 { 0 } else { 1 } + 2 * (src % 8) - 1;
    (src / 8) * 14 + attack
}

fn map_knight_threat(src: usize, dest: usize) -> usize {
    offsets::KNIGHT[src] + below(src, dest, &attacks::KNIGHT)
}

fn map_bishop_threat(src: usize, dest: usize) -> usize {
    offsets::BISHOP[src] + below(src, dest, &attacks::BISHOP)
}

fn map_rook_threat(src: usize, dest: usize) -> usize {
    offsets::ROOK[src] + below(src, dest, &attacks::ROOK)
}

fn map_queen_threat(src: usize, dest: usize) -> usize {
    offsets::QUEEN[src] + below(src, dest, &attacks::QUEEN)
}

fn map_king_threat(src: usize, dest: usize) -> usize {
    offsets::KING[src] + below(src, dest, &attacks::KING)
}
