mod diff;

use bullet_core::{
    graph::{
        builder::{GraphBuilder, InitSettings, Shape},
        Graph, NodeId, NodeIdTy,
    },
    trainer::dataloader::PreparedBatchDevice,
};
use bullet_cuda_backend::CudaDevice;
use montyformat::chess::{Castling, Move, Position};

use crate::{
    data::{loader::prepare, reader::DecompressedData},
    inputs::{INPUT_SIZE, MAX_ACTIVE_BASE, MAX_MOVES},
    moves::NUM_MOVE_INDICES,
};

pub fn make(device: CudaDevice, hl: usize) -> (Graph<CudaDevice>, NodeId) {
    let builder = GraphBuilder::default();

    let inputs = builder.new_sparse_input("inputs", Shape::new(INPUT_SIZE, 1), MAX_ACTIVE_BASE);
    let moves = builder.new_sparse_input("moves", Shape::new(INPUT_SIZE, MAX_MOVES), 4 * MAX_MOVES);
    let targets = builder.new_dense_input("targets", Shape::new(MAX_MOVES, 1));
    let buckets = builder.new_sparse_input("buckets", Shape::new(NUM_MOVE_INDICES, MAX_MOVES), MAX_MOVES);

    let l0 = builder.new_affine("l0", INPUT_SIZE, hl);
    let mw = builder.new_weights("mw", Shape::new(hl, INPUT_SIZE), InitSettings::Normal { mean: 0.0, stdev: 0.01 });
    let l1w =
        builder.new_weights("l1w", Shape::new(hl, NUM_MOVE_INDICES), InitSettings::Normal { mean: 0.0, stdev: 0.01 });

    let base_hl = l0.forward(inputs).crelu();
    let ones = builder.new_constant(Shape::new(1, MAX_MOVES), &[1.0; MAX_MOVES]);

    let logits = builder.apply(diff::ApplyMoveDiffAndDot {
        weights: mw.annotated_node(),
        moves: moves.annotated_node(),
        hl: base_hl.annotated_node(),
        out_weights: l1w.annotated_node(),
        buckets: buckets.annotated_node(),
    });

    let loss = logits.softmax_crossentropy_loss(targets);
    let _ = ones.matmul(loss);

    let node = NodeId::new(loss.annotated_node().idx, NodeIdTy::Ancillary(0));
    (builder.build(device), node)
}

pub fn save_quantised(graph: &Graph<CudaDevice>, path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path).unwrap();

    let mut quant = Vec::new();

    for id in ["l0w", "mw", "l0b", "l1w"] {
        let vals = graph.get_weights(id).get_dense_vals().unwrap();

        for x in vals {
            let q = (x * 128.0).round() as i8;
            assert_eq!((x * 128.0).round(), f32::from(q));
            quant.extend_from_slice(&q.to_le_bytes());
        }
    }

    file.write_all(&quant)
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
