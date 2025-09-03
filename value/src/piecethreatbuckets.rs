use bullet_lib::game::formats::bulletformat::ChessBoard;
use bullet_lib::game::inputs::OutputBuckets;
use bullet_lib::default::inputs::SparseInputType;
use crate::consts::offsets;
use crate::input::ThreatInputs;

#[derive(Clone, Copy, Default)]
pub struct PieceThreatCount;

#[derive(Clone, Copy)]
struct Bucket {
    idx: u8,
    min_piece: u8,
    max_piece: Option<u8>,   // None = +inf
    min_threat: u16,
    max_threat: Option<u16>, // None = +inf
}

const PIECE_THREAT_BUCKETS: [Bucket; 37] = [
    Bucket { idx: 0, min_piece: 0, max_piece: Some(5), min_threat: 0, max_threat: Some(2) },
    Bucket { idx: 1, min_piece: 0, max_piece: Some(5), min_threat: 3, max_threat: None },
    Bucket { idx: 2, min_piece: 6, max_piece: Some(8), min_threat: 0, max_threat: Some(2) },
    Bucket { idx: 3, min_piece: 6, max_piece: Some(8), min_threat: 3, max_threat: Some(5) },
    Bucket { idx: 4, min_piece: 6, max_piece: Some(8), min_threat: 6, max_threat: None },
    Bucket { idx: 5, min_piece: 9, max_piece: Some(11), min_threat: 0, max_threat: Some(2) },
    Bucket { idx: 6, min_piece: 9, max_piece: Some(11), min_threat: 3, max_threat: Some(5) },
    Bucket { idx: 7, min_piece: 9, max_piece: Some(11), min_threat: 6, max_threat: Some(8) },
    Bucket { idx: 8, min_piece: 9, max_piece: Some(11), min_threat: 9, max_threat: None },
    Bucket { idx: 9, min_piece: 12, max_piece: Some(14), min_threat: 0, max_threat: Some(5) },
    Bucket { idx: 10, min_piece: 12, max_piece: Some(14), min_threat: 6, max_threat: Some(9) },
    Bucket { idx: 11, min_piece: 12, max_piece: Some(14), min_threat: 10, max_threat: Some(13) },
    Bucket { idx: 12, min_piece: 12, max_piece: Some(14), min_threat: 14, max_threat: None },
    Bucket { idx: 13, min_piece: 15, max_piece: Some(17), min_threat: 0, max_threat: Some(9) },
    Bucket { idx: 14, min_piece: 15, max_piece: Some(17), min_threat: 10, max_threat: Some(14) },
    Bucket { idx: 15, min_piece: 15, max_piece: Some(17), min_threat: 15, max_threat: Some(19) },
    Bucket { idx: 16, min_piece: 15, max_piece: Some(17), min_threat: 20, max_threat: None },
    Bucket { idx: 17, min_piece: 18, max_piece: Some(20), min_threat: 0, max_threat: Some(14) },
    Bucket { idx: 18, min_piece: 18, max_piece: Some(20), min_threat: 15, max_threat: Some(20) },
    Bucket { idx: 19, min_piece: 18, max_piece: Some(20), min_threat: 21, max_threat: Some(26) },
    Bucket { idx: 20, min_piece: 18, max_piece: Some(20), min_threat: 26, max_threat: None },
    Bucket { idx: 21, min_piece: 21, max_piece: Some(23), min_threat: 0, max_threat: Some(20) },
    Bucket { idx: 22, min_piece: 21, max_piece: Some(23), min_threat: 21, max_threat: Some(26) },
    Bucket { idx: 23, min_piece: 21, max_piece: Some(23), min_threat: 27, max_threat: Some(32) },
    Bucket { idx: 24, min_piece: 21, max_piece: Some(23), min_threat: 33, max_threat: None },
    Bucket { idx: 25, min_piece: 24, max_piece: Some(26), min_threat: 0, max_threat: Some(26) },
    Bucket { idx: 26, min_piece: 24, max_piece: Some(26), min_threat: 27, max_threat: Some(33) },
    Bucket { idx: 27, min_piece: 24, max_piece: Some(26), min_threat: 34, max_threat: Some(40) },
    Bucket { idx: 28, min_piece: 24, max_piece: Some(26), min_threat: 41, max_threat: None },
    Bucket { idx: 29, min_piece: 27, max_piece: Some(29), min_threat: 0, max_threat: Some(32) },
    Bucket { idx: 30, min_piece: 27, max_piece: Some(29), min_threat: 33, max_threat: Some(39) },
    Bucket { idx: 31, min_piece: 27, max_piece: Some(29), min_threat: 40, max_threat: Some(46) },
    Bucket { idx: 32, min_piece: 27, max_piece: Some(29), min_threat: 47, max_threat: None },
    Bucket { idx: 33, min_piece: 30, max_piece: None, min_threat: 0, max_threat: Some(37) },
    Bucket { idx: 34, min_piece: 30, max_piece: None, min_threat: 38, max_threat: Some(44) },
    Bucket { idx: 35, min_piece: 30, max_piece: None, min_threat: 45, max_threat: Some(51) },
    Bucket { idx: 36, min_piece: 30, max_piece: None, min_threat: 52, max_threat: None },
];

impl OutputBuckets<ChessBoard> for PieceThreatCount {
    const BUCKETS: usize = 37;
    #[inline]
    fn bucket(&self, pos: &ChessBoard) -> u8 {
        const TOTAL_THREATS: usize = 2 * offsets::END;
        let inputs = ThreatInputs;
        let mut pieces: u16 = 0;
        let mut threats: u16 = 0;
        inputs.map_features(pos, |idx, _| {
            if idx < TOTAL_THREATS {
                threats += 1;
            } else {
                pieces += 1;
            }
        });

        let p = pieces as u8;
        let t = threats as u16;
        for b in &PIECE_THREAT_BUCKETS {
            let piece_ok = p >= b.min_piece && b.max_piece.map(|mp| p <= mp).unwrap_or(true);
            let threat_ok = t >= b.min_threat && b.max_threat.map(|mt| t <= mt).unwrap_or(true);
            if piece_ok && threat_ok {
                return b.idx;
            }
        }
        36 // fallback (should be unreachable)
    }
}