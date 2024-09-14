use montyformat::chess::{Position, Piece, Side};
use tch::{nn, Kind, Tensor};

pub const INPUTS: i64 = 768;
pub const EMBED_SIZE: i64 = 16;
pub const FROM_SUBNETS: i64 = 64;
pub const DEST_SUBNETS: i64 = 64;
pub const OUTPUTS: i64 = FROM_SUBNETS * DEST_SUBNETS;

#[derive(Debug)]
pub struct PolicyNetwork {
    from_subnet_l1: nn::Linear,
    dest_subnet_l1: nn::Linear,
}

impl PolicyNetwork {
    pub fn new(vs: &nn::Path) -> Self {
        const FROM_HL: i64 = EMBED_SIZE * FROM_SUBNETS;
        const DEST_HL: i64 = EMBED_SIZE * DEST_SUBNETS;

        Self {
            from_subnet_l1: nn::linear(vs, INPUTS, FROM_HL, Default::default()),
            dest_subnet_l1: nn::linear(vs, INPUTS, DEST_HL, Default::default()),
        }
    }

    pub fn forward(&self, xs: &Tensor, legal_mask: &Tensor, batch_size: i64) -> Tensor {
        println!("froms");
        let froms = xs.apply(&self.from_subnet_l1).relu().reshape([batch_size, FROM_SUBNETS, EMBED_SIZE]);

        println!("dests");
        let dests = xs.apply(&self.dest_subnet_l1).relu().reshape([batch_size, EMBED_SIZE, DEST_SUBNETS]);

        println!("{:?}", froms.size());
        println!("{:?}", dests.size());

        println!("logits");
        let raw_logits = froms.matmul(&dests);
        println!("{:?}", raw_logits.size());
        let raw_logits = raw_logits.reshape([batch_size, FROM_SUBNETS * DEST_SUBNETS]);

        println!("masks");
        let masked = raw_logits.masked_fill(legal_mask, f64::NEG_INFINITY);

        println!("softmax");
        masked.softmax(1, Kind::Float)
    }
}

pub fn map_policy_inputs<F: FnMut(usize)>(pos: &Position, mut f: F) {
    let flip = pos.stm() == Side::BLACK;

    for piece in Piece::PAWN..=Piece::KING {
        let pc = 64 * (piece - 2);

        let mut our_bb = pos.piece(piece) & pos.piece(pos.stm());
        let mut opp_bb = pos.piece(piece) & pos.piece(pos.stm() ^ 1);

        if flip {
            our_bb = our_bb.swap_bytes();
            opp_bb = opp_bb.swap_bytes();
        }

        while our_bb > 0 {
            f(pc + our_bb.trailing_zeros() as usize);
            our_bb &= our_bb - 1;
        }

        while opp_bb > 0 {
            f(384 + pc + opp_bb.trailing_zeros() as usize);
            opp_bb &= opp_bb - 1;
        }
    }
}