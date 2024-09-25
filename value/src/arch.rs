use montyformat::chess::{Castling, Piece, Position};
use tch::{
    nn, Device, Kind, Tensor
};

use crate::{loader::{DataLoader, PreAllocs}, save::SavedNetworkFormat};

pub const INPUTS: i64 = 256;
pub const TOKENS: i64 = 12;
pub const DK: i64 = 32;
pub const DV: i64 = 32;

struct OutputHead {
    l1: nn::Linear,
    l2: nn::Linear,
}

impl OutputHead {
    fn new(vs: &nn::Path) -> Self {
        Self {
            l1: nn::linear(vs, DV * TOKENS, 16, Default::default()),
            l2: nn::linear(vs, 16, 1, Default::default()),
        }
    }

    fn fwd(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.l1).relu().apply(&self.l2)
    }
}

struct QKVEmbedding {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
}

impl QKVEmbedding {
    fn new(vs: &nn::Path) -> Self {
        let config = nn::LinearConfig {
            bias: false,
            bs_init: None,
            ..Default::default()
        };

        Self {
            q: nn::linear(vs, INPUTS, DK, config),
            k: nn::linear(vs, INPUTS, DK, config),
            v: nn::linear(vs, INPUTS, DV, config),
        }
    }
}

pub struct ValueNetwork {
    qkvs: Vec<QKVEmbedding>,
    out: OutputHead,
}

impl ValueNetwork {
    pub fn randomised(vs: &nn::Path) -> Self {
        let mut net = Self {
            qkvs: Vec::new(),
            out: OutputHead::new(vs),
        };

        for _ in 0..TOKENS {
            net.qkvs.push(QKVEmbedding::new(vs));
        }

        net
    }

    pub fn fwd(&self, xs: &[Tensor], batch_size: i64) -> Tensor {
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for (qkv, x) in self.qkvs.iter().zip(xs.iter()) {
            queries.push(qkv.q.ws.matmul(x).transpose(-2, -1));
            keys.push(qkv.k.ws.matmul(x).transpose(-2, -1));
            values.push(qkv.v.ws.matmul(x).transpose(-2, -1));
        }

        let query = Tensor::concat(&queries, -1).reshape([batch_size, TOKENS, DK]);
        let key = Tensor::concat(&keys, -1).reshape([batch_size, TOKENS, DK]);
        let value = Tensor::concat(&values, -1).reshape([batch_size, TOKENS, DV]);

        let scale = 1.0 / (*query.size().last().unwrap() as f32).sqrt();
        let dots = scale * query.bmm(&key.transpose(-2, -1));
        let softmaxed = dots.softmax(-1, Kind::Float);
        let attention = softmaxed.bmm(&value).reshape([batch_size, TOKENS * DV]);

        self.out.fwd(&attention.relu())
    }

    pub fn export(&self) -> Box<SavedNetworkFormat> {
        let mut net = SavedNetworkFormat::boxed_and_zeroed();

        for (i, qkv) in self.qkvs.iter().enumerate() {
            SavedNetworkFormat::write_linear_into_matrix(&qkv.q, &mut net.wq[i]);
            SavedNetworkFormat::write_linear_into_matrix(&qkv.k, &mut net.wk[i]);
            SavedNetworkFormat::write_linear_into_matrix(&qkv.v, &mut net.wv[i]);
        }

        SavedNetworkFormat::write_linear_into_layer(&self.out.l1, &mut net.l1);
        SavedNetworkFormat::write_linear_into_layer(&self.out.l2, &mut net.l2);

        net
    }

    pub fn run_sample_fens(&self) {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        ];

        let mut positions = Vec::new();
        let mut castling = Castling::default();

        for fen in fens {
            positions.push((Position::parse_fen(fen, &mut castling), 0.5));
        }

        let device = Device::cuda_if_available();
        let mut preallocs = PreAllocs::new(fens.len());

        let inputs = DataLoader::get_batch_inputs(device, &positions, &mut preallocs);
        let outputs = self.fwd(&inputs.0, fens.len() as i64);

        let mut buffer = vec![0f32; fens.len()];
        outputs.copy_data(&mut buffer, fens.len());
        for (fen, score) in fens.iter().zip(buffer.iter()) {
            println!("FEN: {fen}");
            println!("EVAL: {:.0}cp", 400.0 * score)
        }
    }
}

pub fn map_value_features<F: FnMut(usize, usize)>(pos: &Position, mut f: F) {
    let threats = pos.threats_by(1 - pos.stm());
    let defences = pos.threats_by(pos.stm());

    let flip = if pos.stm() > 0 { 56 } else { 0 };

    for (stm, &side) in [pos.stm(), 1 - pos.stm()].iter().enumerate() {
        for piece in Piece::PAWN..=Piece::KING {
            let piece_idx = 6 * stm + piece - 2;

            let mut bb = pos.piece(side) & pos.piece(piece);
            while bb > 0 {
                let sq = bb.trailing_zeros() as usize;

                let bit = 1 << sq;
                let state = usize::from(bit & threats > 0) + 2 * usize::from(bit & defences > 0);

                f(piece_idx, 64 * state + (sq ^ flip));

                bb &= bb - 1;
            }
        }
    }
}
