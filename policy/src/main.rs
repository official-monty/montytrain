mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    nn::{
        optimiser::{AdamWParams, Optimiser}, Activation, ExecutionContext, Graph, NetworkBuilder, Node, Shape
    },
    trainer::{
        logger,
        save::{Layout, QuantTarget, SavedFormat},
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
        NetworkTrainer,
    },
};

use trainer::Trainer;

const ID: &str = "policy001";

fn main() {
    let data_preparer = preparer::DataPreparer::new("data/policygen6.binpack", 4096);

    let size = 128;

    let (graph, output_node) = network(size);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: Optimiser::new(graph, optimiser_params).unwrap(),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 1,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.00001,
            final_superbatch: 600,
        },
        save_rate: 40,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                trainer
                    .save_weights_portion(
                        &format!("checkpoints/{ID}-{sb}.network"),
                        &[
                            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new(
                                "l1w",
                                QuantTarget::Float,
                                Layout::Transposed(Shape::new(moves::NUM_MOVES, 2 * size)),
                            ),
                            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        eval_fen(&mut trainer, output_node, fen);
    }
}

fn network(size: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    let stm = builder.new_sparse_input("stm", Shape::new(inputs::INPUT_SIZE, 1), inputs::MAX_ACTIVE);
    let ntm = builder.new_sparse_input("ntm", Shape::new(inputs::INPUT_SIZE, 1), inputs::MAX_ACTIVE);
    let mask = builder.new_sparse_input("mask", Shape::new(moves::NUM_MOVES, 1), moves::MAX_MOVES);
    let dist = builder.new_dense_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", 2 * size, moves::NUM_MOVES);

    let mut out = l0.forward_sparse_dual_with_activation(stm, ntm, Activation::CReLU);
    out = l1.forward(out);
    out.masked_softmax_crossentropy_loss(dist, mask);

    let out = out.node();
    (builder.build(ExecutionContext::default()), out)
}

fn eval_fen(trainer: &mut Trainer, output_node: Node, fen: &str) {
    use bullet::trainer::default::formats::montyformat::chess::{Position, Castling, Move};

    let mut castling = Castling::default();
    let pos = Position::parse_fen(fen, &mut castling);
    let mut data = loader::DecompressedData { pos, moves: [(0, 0); 108], num: 0 };

    pos.map_legal_moves(&castling, |mov| {
        data.moves[data.num] = (u16::from(mov), 1);
        data.num += 1;
    });

    let prepared = preparer::PreparedData::new(&[data], 1);

    trainer.load_batch(&prepared);

    trainer.optimiser.graph.forward().unwrap();

    let output = trainer.optimiser.graph.get_node(output_node).get_dense_vals().unwrap();

    println!("FEN: {fen}");
    for (mov, _) in &data.moves[..data.num] {
        let mov = Move::from(*mov);
        println!("{}: {}", mov.to_uci(&castling), output[moves::map_move_to_index(&pos, mov)]);
    }
}
