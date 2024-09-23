use montyformat::chess::{Move, Piece, Position, Side};
use tch::{nn, Tensor};

pub const INPUTS: i64 = 768;
pub const EMBED_SIZE: i64 = 16;
pub const FROM_SUBNETS: i64 = 64;
pub const DEST_SUBNETS: i64 = 64;
pub const OUTPUTS: i64 = FROM_SUBNETS * DEST_SUBNETS;

#[derive(Debug)]
pub struct SubNet<const SUBNETS: i64> {
    l1: nn::Linear,
}

impl<const SUBNETS: i64> SubNet<SUBNETS> {
    fn new(vs: &nn::Path) -> Self {
        let hl = EMBED_SIZE * SUBNETS;

        Self {
            l1: nn::linear(vs, INPUTS, hl, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor, _batch_size: i64) -> Tensor {
        xs.apply(&self.l1).relu()
    }
}

#[derive(Debug)]
pub struct PolicyNetwork {
    from_subnets: SubNet<FROM_SUBNETS>,
    dest_subnets: SubNet<DEST_SUBNETS>,
}

impl PolicyNetwork {
    pub fn new(vs: &nn::Path) -> Self {
        Self {
            from_subnets: SubNet::new(vs),
            dest_subnets: SubNet::new(vs),
        }
    }

    pub fn forward_raw(&self, xs: &Tensor, batch_size: i64) -> Tensor {
        let froms = self.from_subnets.forward(xs, batch_size);
        let dests = self.dest_subnets.forward(xs, batch_size);

        attention(&froms, &dests, batch_size)
    }
}

fn attention(froms: &Tensor, dests: &Tensor, batch_size: i64) -> Tensor {
    let froms = froms.reshape([batch_size, FROM_SUBNETS, EMBED_SIZE]);
    let dests = dests.reshape([batch_size, EMBED_SIZE, FROM_SUBNETS]);

    froms
        .matmul(&dests)
        .reshape([batch_size, FROM_SUBNETS * DEST_SUBNETS])
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

pub fn map_move_to_index(pos: &Position, mov: Move) -> usize {
    let flip = if pos.stm() == Side::BLACK { 56 } else { 0 };
    usize::from(64 * (mov.src() ^ flip) + (mov.to() ^ flip))
}
