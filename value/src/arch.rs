use montyformat::chess::{Piece, Position};
use tch::{
    nn,
    Kind, Tensor,
};

use crate::save::SavedNetworkFormat;

pub const INPUTS: i64 = 256;
pub const TOKENS: i64 = 12;
pub const DK: i64 = 32;
pub const DV: i64 = 8;

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
