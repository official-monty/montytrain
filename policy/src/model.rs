mod grab;

use bullet_core::{
    graph::{
        builder::{GraphBuilder, Shape},
        Graph, NodeId, NodeIdTy,
    },
    trainer::dataloader::PreparedBatchDevice,
};
use bullet_cuda_backend::CudaDevice;
use montyformat::chess::{Castling, Move, Position};

use crate::{
    data::{loader::prepare, reader::DecompressedData},
    inputs::{INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES},
};

pub fn make(device: CudaDevice, src_hl: usize, dst_hl: usize, fact_hl: usize, dim: usize) -> (Graph<CudaDevice>, NodeId) {
    let builder = GraphBuilder::default();

    let inputs = builder.new_sparse_input("inputs", Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE);
    let targets = builder.new_dense_input("targets", Shape::new(MAX_MOVES, 1));
    let moves = builder.new_sparse_input("moves", Shape::new(64, 128), MAX_MOVES);
    let moves2 = builder.new_sparse_input("moves2", Shape::new(1880 * 2, 1), MAX_MOVES);

    let subnets = |name: &str, num, hl| {
        let l0 = builder.new_affine(&format!("{name}0"), INPUT_SIZE, hl);
        let l1 = builder.new_affine(&format!("{name}1"), hl / 2, num * dim);
        let hl = l0.forward(inputs).crelu().pairwise_mul();
        l1.forward(hl).reshape(Shape::new(dim, num))
    };

    let attn = subnets("src", 64, src_hl).gemm(true, subnets("dst", 128, dst_hl), false);
    
    let fact0 = builder.new_affine("fact0", INPUT_SIZE, fact_hl);
    let fact1 = builder.new_affine("fact0", fact_hl / 2, 1880 * 2);

    let facthl = fact0.forward(inputs).crelu().pairwise_mul();
    let fact = fact1.forward(facthl);

    let logits = builder.apply(grab::Grab::new(attn, moves)) + builder.apply(grab::Grab::new(fact, moves2));

    let ones = builder.new_constant(Shape::new(1, MAX_MOVES), &[1.0; MAX_MOVES]);
    let loss = logits.softmax_crossentropy_loss(targets);
    let _ = ones.matmul(loss);

    let node = NodeId::new(loss.annotated_node().idx, NodeIdTy::Ancillary(0));
    (builder.build(device), node)
}

pub fn eval(graph: &mut Graph<CudaDevice>, node: NodeId, fen: &str) {
    let mut castling = Castling::default();
    let pos = Position::parse_fen(fen, &mut castling);

    let mut moves = [(0, 0); 64];
    let mut num = 0;

    pos.map_legal_moves(&castling, |mov| {
        moves[num] = (u16::from(mov), 1);
        num += 1;
    });

    let point = DecompressedData { pos, castling, moves, num };

    let data = prepare(&[point], 1);

    let mut on_device = PreparedBatchDevice::new(graph.device(), &data).unwrap();

    on_device.load_into_graph(graph).unwrap();

    let _ = graph.forward().unwrap();

    let dist = graph.get(node).unwrap().get_dense_vals().unwrap();

    println!();
    println!("{fen}");
    for i in 0..num {
        println!("{} -> {:.2}%", Move::from(moves[i].0).to_uci(&castling), dist[i] * 100.0)
    }
}

pub fn save_quantised(graph: &Graph<CudaDevice>, path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path).unwrap();

    let mut quant = Vec::new();

    for id in ["src0w", "src0b", "dst0w", "dst0b"] {
        let vals = graph.get_weights(id).get_dense_vals().unwrap();

        for x in vals {
            let q = (x * 255.0).round() as i16;
            assert_eq!((x * 255.0).round(), f32::from(q));
            quant.extend_from_slice(&q.to_le_bytes());
        }
    }

    for id in ["src1w", "dst1w"] {
        let vals = graph.get_weights(id).get_dense_vals().unwrap();

        for x in vals {
            let q = (x * 64.0).round() as i8;
            assert_eq!((x * 64.0).round(), f32::from(q));
            quant.extend_from_slice(&q.to_le_bytes());
        }
    }

    for id in ["src1b", "dst1b"] {
        let vals = graph.get_weights(id).get_dense_vals().unwrap();

        for x in vals {
            let q = (x * 255.0 * 64.0).round() as i16;
            assert_eq!((x * 255.0 * 64.0).round(), f32::from(q));
            quant.extend_from_slice(&q.to_le_bytes());
        }
    }

    file.write_all(&quant)
}
