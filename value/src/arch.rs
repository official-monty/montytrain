use montyformat::chess::{Move, Piece, Position, Side};
use tch::{
    nn::{self, Module},
    Kind, Tensor,
};

pub const TOKENS: i64 = 12;
const DK: i64 = 32;
const DV: i64 = 8;

struct OutputHead {
    l1: nn::Linear,
    l2: nn::Linear,
}

impl OutputHead {
    fn new(vs: &nn::Path) -> Self {
        Self {
            l1: nn::linear(vs, DV * TOKENS, 16, Default::default()),
            l2: nn::linear(vs, DV * 16, 1, Default::default()),
        }
    }

    fn fwd(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.l1).relu().apply(&self.l2)
    }
}

struct QKV {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
}

impl QKV {
    fn new(vs: &nn::Path) -> Self {
        let config = nn::LinearConfig {
            bias: false,
            bs_init: None,
            ..Default::default()
        };

        Self {
            q: nn::linear(vs, 256, DK, config),
            k: nn::linear(vs, 256, DK, config),
            v: nn::linear(vs, 256, DV, config),
        }
    }
}

pub struct ValueNetwork {
    qkvs: Vec<QKV>,
    out: OutputHead,
}

impl ValueNetwork {
    pub fn new(vs: &nn::Path) -> Self {
        let mut net = Self {
            qkvs: Vec::new(),
            out: OutputHead::new(vs),
        };

        for _ in 0..TOKENS {
            net.qkvs.push(QKV::new(vs));
        }

        net
    }

    pub fn fwd(&self, xs: &Tensor) -> Tensor {
        let mut queries = Vec::new();
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for qkv in &self.qkvs {
            queries.push(qkv.q.forward(xs));
            keys.push(qkv.k.forward(xs));
            values.push(qkv.v.forward(xs));
        }

        let query = Tensor::concat(&queries, -2);
        let key = Tensor::concat(&keys, -2);
        let value = Tensor::concat(&values, -2);

        let scale = 1.0 / (*query.size().last().unwrap() as f32).sqrt();
        let dots = scale * query.bmm(&key.transpose(-2, -1));
        let softmaxed = dots.softmax(-1, Kind::Float);
        let attention = softmaxed.bmm(&value);

        self.out.fwd(&attention.relu())
    }
}
