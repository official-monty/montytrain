mod select_affine;

use bullet_core::{
    graph::{
        builder::{GraphBuilder, Shape},
        Graph, GraphNodeId, GraphNodeIdTy,
    },
    trainer::dataloader::PreparedBatchDevice,
};
use bullet_cuda_backend::CudaDevice;
use montyformat::chess::{Castling, Move, Position};

use crate::{
    data::{loader::prepare, reader::DecompressedData},
    inputs::{INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES, NUM_MOVES_INDICES},
};

pub fn make(device: CudaDevice, hl: usize) -> (Graph<CudaDevice>, GraphNodeId) {
    let builder = GraphBuilder::default();

    let inputs = builder.new_sparse_input("inputs", Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE);
    let targets = builder.new_dense_input("targets", Shape::new(MAX_MOVES, 1));
    let moves = builder.new_sparse_input("moves", Shape::new(NUM_MOVES_INDICES, 1), MAX_MOVES);

    let l0 = builder.new_affine("l0", INPUT_SIZE, hl);
    let l1 = builder.new_affine("l1", hl / 2, NUM_MOVES_INDICES);

    let hl = l0.forward(inputs).crelu().pairwise_mul();

    let logits = builder.apply(select_affine::SelectAffine::new(l1, hl, moves));

    let ones = builder.new_constant(Shape::new(1, MAX_MOVES), &[1.0; MAX_MOVES]);
    let loss = logits.softmax_crossentropy_loss(targets);
    let _ = ones.matmul(loss);

    let node = GraphNodeId::new(loss.annotated_node().idx, GraphNodeIdTy::Ancillary(0));
    (builder.build(device), node)
}

pub fn eval(graph: &mut Graph<CudaDevice>, node: GraphNodeId, fen: &str) {
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

    for id in ["l0w", "l0b", "l1w", "l1b"] {
        let vals = graph.get_weights(id).get_dense_vals().unwrap();

        for x in vals {
            let q = (x * 128.0).round() as i8;
            assert_eq!((x * 128.0).round(), f32::from(q));
            quant.extend_from_slice(&q.to_le_bytes());
        }
    }

    file.write_all(&quant)
}
