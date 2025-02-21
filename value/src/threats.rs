use bullet::default::formats::montyformat::chess::Piece;

use crate::consts::{attacks, indices, offsets};

pub fn map_piece_threat(
    piece: usize,
    src: usize,
    dest: usize,
    enemy: bool,
) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest, enemy),
        Piece::KNIGHT => map_knight_threat(src, dest, enemy),
        Piece::BISHOP => map_bishop_threat(src, dest, enemy),
        Piece::ROOK => map_rook_threat(src, dest, enemy),
        Piece::QUEEN => map_queen_threat(src, dest, enemy),
        Piece::KING => map_king_threat(src, dest, enemy),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_pawn_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let up = usize::from(dest > src);
    let diff = dest.abs_diff(src);
    let id = if diff == [9, 7][up] { 0 } else { 1 };
    let attack = 2 * (src % 8) + id - 1;
    let threat = offsets::PAWN + usize::from(enemy) * indices::PAWN + (src / 8 - 1) * 14 + attack;

    assert!(threat < offsets::KNIGHT, "{threat}");

    Some(threat)
}

fn map_knight_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
    let threat = offsets::KNIGHT + usize::from(enemy) * indices::KNIGHT[64] + idx;

    assert!(threat >= offsets::KNIGHT, "{threat}");
    assert!(threat < offsets::BISHOP, "{threat}");

    Some(threat)
}

fn map_bishop_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
    let threat = offsets::BISHOP + usize::from(enemy) * indices::BISHOP[64] + idx;

    assert!(threat >= offsets::BISHOP, "{threat}");
    assert!(threat < offsets::ROOK, "{threat}");

    Some(threat)
}

fn map_rook_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
    let threat = offsets::ROOK + usize::from(enemy) * indices::ROOK[64] + idx;

    assert!(threat >= offsets::ROOK, "{threat}");
    assert!(threat < offsets::QUEEN, "{threat}");

    Some(threat)
}

fn map_queen_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
    let threat = offsets::QUEEN + usize::from(enemy) * indices::QUEEN[64] + idx;

    assert!(threat >= offsets::QUEEN, "{threat}");
    assert!(threat < offsets::KING, "{threat}");

    Some(threat)
}

fn map_king_threat(src: usize, dest: usize, enemy: bool) -> Option<usize> {
    let idx = indices::KING[src] + below(src, dest, &attacks::KING);
    let threat = offsets::KING + usize::from(enemy) * indices::KING[64] + idx;

    assert!(threat >= offsets::KING, "{threat}");
    assert!(threat < offsets::END, "{threat}");

    Some(threat)
}
