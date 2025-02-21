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
        Piece::KNIGHT => map_threat(src, dest, enemy, &indices::KNIGHT, &attacks::KNIGHT, offsets::KNIGHT),
        Piece::BISHOP => map_threat(src, dest, enemy, &indices::BISHOP, &attacks::BISHOP, offsets::BISHOP),
        Piece::ROOK => map_threat(src, dest, enemy, &indices::ROOK, &attacks::ROOK, offsets::ROOK),
        Piece::QUEEN => map_threat(src, dest, enemy, &indices::QUEEN, &attacks::QUEEN, offsets::QUEEN),
        Piece::KING => map_threat(src, dest, enemy, &indices::KING, &attacks::KING, offsets::KING),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_threat(src: usize, dest: usize, enemy: bool, indices: &[usize; 65], attacks: &[u64; 64], offset: usize) -> Option<usize> {
    let idx = indices[src] + below(src, dest, attacks);
    let threat = offset + usize::from(enemy) * indices[64] + idx;

    assert!(threat >= offset);
    assert!(threat < offsets::END);

    Some(threat)
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
