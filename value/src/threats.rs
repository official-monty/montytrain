use montyformat::chess::Piece;

use crate::consts::{attacks, offsets};

pub fn map_piece_threat(piece: usize, src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest, target, enemy),
        Piece::KNIGHT => map_knight_threat(src, dest, target, enemy),
        Piece::BISHOP => map_bishop_threat(src, dest, target, enemy),
        Piece::ROOK => map_rook_threat(src, dest, target, enemy),
        Piece::QUEEN => map_queen_threat(src, dest, target, enemy),
        Piece::KING => map_king_threat(src, dest, target, enemy),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_pawn_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    if [Piece::BISHOP, Piece::QUEEN, Piece::KING].contains(&target) {
        None
    } else if target == Piece::PAWN && enemy && dest > src {
        None
    } else {
        let diff = if dest > src { dest - src } else { src - dest };
        let attack = if diff == 7 { 0 } else { 1 } + 2 * (src % 8) - 1;
        Some((src / 8) * 14 + attack)
    }
}

fn map_knight_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    Some(offsets::KNIGHT[src] + below(src, dest, &attacks::KNIGHT))
}

fn map_bishop_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    Some(offsets::BISHOP[src] + below(src, dest, &attacks::BISHOP))
}

fn map_rook_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    Some(offsets::ROOK[src] + below(src, dest, &attacks::ROOK))
}

fn map_queen_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    Some(offsets::QUEEN[src] + below(src, dest, &attacks::QUEEN))
}

fn map_king_threat(src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    Some(offsets::KING[src] + below(src, dest, &attacks::KING))
}
