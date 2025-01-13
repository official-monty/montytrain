mod consts;
mod preparer;
mod threat_mask;
mod threat_inputs;
mod trainer;

use bullet::{
    format, logger, lr,
    montyformat::chess::{Move, Position},
    operations,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    save::{Layout, SavedFormat},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Node, QuantTarget, Shape,
    TrainingSchedule, TrainingSteps,
};
use trainer::Trainer;

const ID: &str = "threat-mask-subnet";
const SIZE: usize = 2048;
const OUT_DIM: usize = 16;

fn main() {
    let data_preparer = preparer::DataPreparer::new(
        "/home/privateclient/monty_value_training/interleaved.binpack",
        96000,
        8,
        |_, _, _, _| true,
    );

    //let data_preparer = preparer::DataPreparer::new("data/datagen19.binpack", 4096, 4, |_, _, _, _| true);

    let (mut graph, output_node) = network();

    let post_embed_stdev = 1.0 / ((SIZE / 2) as f32).sqrt();

    graph.get_weights_mut("embw").seed_random(0.0, 1.0 / (threat_inputs::INPUT_SIZE as f32).sqrt(), true);

    graph.get_weights_mut("l1w").seed_random(0.0, post_embed_stdev, true);
    graph.get_weights_mut("l2w").seed_random(0.0, 1.0 / 16f32.sqrt(), true);

    graph.get_weights_mut("s1w").seed_random(0.0, post_embed_stdev, true);
    graph.get_weights_mut("s2w").seed_random(0.0, 1.0 / 256f32.sqrt(), true);

    graph.get_weights_mut("n1w").seed_random(0.0, post_embed_stdev, true);
    graph.get_weights_mut("n2w").seed_random(0.0, 1.0 / 256f32.sqrt(), true);

    let optimiser_params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -0.99, max_weight: 0.99 };

    let mut trainer = Trainer { optimiser: AdamWOptimiser::new(graph, optimiser_params) };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 128,
            start_superbatch: 1,
            end_superbatch: 3000,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 1.0 },
        lr_scheduler: lr::ExponentialDecayLR { initial_lr: 0.001, final_lr: 0.0000001, final_superbatch: 3000 },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 4, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    logger::clear_colours();
    println!("{}", logger::ansi("Beginning Training", "34;1"));
    schedule.display();
    settings.display();

    trainer.train_custom(
        &data_preparer,
        &Option::<preparer::DataPreparer<fn(&Position, Move, i16, f32) -> bool>>::None,
        &schedule,
        &settings,
        |sb, trainer, schedule, _| {
            if schedule.should_save(sb) {
                save_quantised(trainer, &format!("checkpoints/{ID}-{sb}/quantised.network"));
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
        eval(&mut trainer, output_node, fen)
    }
}

fn network() -> (Graph, Node) {
    let builder = &mut GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(threat_inputs::INPUT_SIZE, 1));
    let stm_mask = builder.create_input("stm_mask", Shape::new(threat_mask::NUM_THREATS, 1));
    let ntm_mask = builder.create_input("ntm_mask", Shape::new(threat_mask::NUM_THREATS, 1));
    let target = builder.create_input("target", Shape::new(3, 1));

    let embedding = embedding(builder, inputs);

    let main_subnet = main_subnet(builder, embedding);

    let stm_threat_subnet = threat_subnet(builder, embedding, stm_mask, 's');
    let ntm_threat_subnet = threat_subnet(builder, embedding, ntm_mask, 'n');
    let threat_subnet = operations::add(builder, stm_threat_subnet, ntm_threat_subnet);

    let psqtw = builder.create_weights("psqt", Shape::new(3, threat_inputs::INPUT_SIZE));
    let output_bias = builder.create_weights("ob", Shape::new(3, 1));
    let dot_prod = operations::submatrix_product(builder, OUT_DIM, main_subnet, threat_subnet);
    let psqt = operations::affine(builder, psqtw, inputs, output_bias);
    let predicted = operations::add(builder, dot_prod, psqt);
    operations::softmax_crossentropy_loss(builder, predicted, target);

    let ctx = ExecutionContext::default();
    (builder.build(ctx), predicted)
}

fn embedding(builder: &mut GraphBuilder, inputs: Node) -> Node {
    let l0w = builder.create_weights("embw", Shape::new(SIZE, threat_inputs::INPUT_SIZE));
    let l0b = builder.create_weights("embb", Shape::new(SIZE, 1));

    let l1 = operations::affine(builder, l0w, inputs, l0b);
    let l1 = operations::activate(builder, l1, Activation::CReLU);
    operations::pairwise_mul(builder, l1)
}

fn main_subnet(builder: &mut GraphBuilder, inputs: Node) -> Node {
    let l1w = builder.create_weights("l1w", Shape::new(OUT_DIM, SIZE / 2));
    let l1b = builder.create_weights("l1b", Shape::new(OUT_DIM, 1));
    let l2w = builder.create_weights("l2w", Shape::new(3 * OUT_DIM, OUT_DIM));
    let l2b = builder.create_weights("l2b", Shape::new(3 * OUT_DIM, 1));

    let l2 = operations::affine(builder, l1w, inputs, l1b);
    let l2 = operations::activate(builder, l2, Activation::SCReLU);
    operations::affine(builder, l2w, l2, l2b)
}

fn threat_subnet(builder: &mut GraphBuilder, inputs: Node, masks: Node, side: char) -> Node {
    let l1w = builder.create_weights(&format!("{side}1w"), Shape::new(threat_mask::NUM_THREATS, SIZE / 2));
    let l1b = builder.create_weights(&format!("{side}1b"), Shape::new(threat_mask::NUM_THREATS, 1));
    let l2w = builder.create_weights(&format!("{side}2w"), Shape::new(OUT_DIM, threat_mask::NUM_THREATS));
    let l2b = builder.create_weights(&format!("{side}2b"), Shape::new(OUT_DIM, 1));

    let out = operations::affine(builder, l1w, inputs, l1b);
    let out = operations::activate(builder, out, Activation::SCReLU);
    let out = operations::mask(builder, out, masks);
    operations::affine(builder, l2w, out, l2b)
}

fn save_quantised(trainer: &Trainer, name: &str) {
    trainer
        .save_weights_portion(
            name,
            &[
                SavedFormat::new("psqt", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("embw", QuantTarget::I16(512), Layout::Normal),
                SavedFormat::new("embb", QuantTarget::I16(512), Layout::Normal),
                SavedFormat::new("l1w", QuantTarget::I16(1024), Layout::Transposed),
                SavedFormat::new("l1b", QuantTarget::I16(1024), Layout::Normal),
                SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("s1w", QuantTarget::I16(1024), Layout::Transposed),
                SavedFormat::new("s1b", QuantTarget::I16(1024), Layout::Normal),
                SavedFormat::new("s2w", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("s2b", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("n1w", QuantTarget::I16(1024), Layout::Transposed),
                SavedFormat::new("n1b", QuantTarget::I16(1024), Layout::Normal),
                SavedFormat::new("n2w", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("n2b", QuantTarget::Float, Layout::Normal),
                SavedFormat::new("ob", QuantTarget::Float, Layout::Normal),
            ],
        )
        .unwrap();
}

fn eval(trainer: &mut Trainer, output_node: Node, fen: &str) {
    let pos = format!("{fen} | 0 | 0.0").parse::<format::ChessBoard>().unwrap();

    let prepared = preparer::PreparedData::new(&[pos], 1);

    trainer.load_batch(&prepared);
    trainer.optimiser.graph_mut().forward();

    let eval = trainer.optimiser.graph().get_node(output_node);

    let vals = eval.get_dense_vals().unwrap();

    let mut win = vals[2];
    let mut draw = vals[1];
    let mut loss = vals[0];

    let max = win.max(draw).max(loss);
    win = (win - max).exp();
    draw = (draw - max).exp();
    loss = (loss - max).exp();
    let sum = win + draw + loss;
    win *= 100.0 / sum;
    draw *= 100.0 / sum;
    loss *= 100.0 / sum;

    println!("FEN: {fen}");
    println!("EVAL: W={:.2}%, D={:.2}%, L={:.2}%", win, draw, loss);
}
