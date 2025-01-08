mod inputs;
mod preparer;
mod threats;
mod trainer;

use bullet::{
    logger, lr,
    montyformat::chess::{Move, Position},
    operations,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    save::{Layout, SavedFormat},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Node, QuantTarget, Shape,
    TrainingSchedule, TrainingSteps,
};
use trainer::Trainer;

const ID: &str = "value001";
const SIZE: usize = 512;
const OUT_DIM: usize = 16;

fn main() {
    //let data_preparer = preparer::DataPreparer::new(
    //    "/home/privateclient/monty_value_training/interleaved.binpack",
    //    96000,
    //    4,
    //    |_, _, _, _| true,
    //);

    let data_preparer = preparer::DataPreparer::new("data/datagen19.binpack", 4096, 4, |_, _, _, _| true);

    let mut graph = network();

    let post_embed_stdev = 1.0 / ((SIZE / 2) as f32).sqrt();

    graph.get_weights_mut("embw").seed_random(0.0, 1.0 / (inputs::INPUT_SIZE as f32).sqrt(), true);

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
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 600,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR { initial_lr: 0.001, final_lr: 0.00001, final_superbatch: 600 },
        save_rate: 40,
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
                trainer
                    .save_weights_portion(
                        &format!("checkpoints/{ID}-{sb}.network"),
                        &[
                            SavedFormat::new("embw", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("embb", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l1w", QuantTarget::Float, Layout::Transposed),
                            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("s1w", QuantTarget::Float, Layout::Transposed),
                            SavedFormat::new("s1b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("s2w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("s2b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("n1w", QuantTarget::Float, Layout::Transposed),
                            SavedFormat::new("n1b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("n2w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("n2b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("ob", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );
}

fn network() -> Graph {
    let builder = &mut GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let stm_mask = builder.create_input("stm_mask", Shape::new(threats::NUM_THREATS, 1));
    let ntm_mask = builder.create_input("ntm_mask", Shape::new(threats::NUM_THREATS, 1));
    let target = builder.create_input("target", Shape::new(3, 1));

    let embedding = embedding(builder, inputs);

    let main_subnet = main_subnet(builder, embedding);

    let stm_threat_subnet = threat_subnet(builder, embedding, stm_mask, 's');
    let ntm_threat_subnet = threat_subnet(builder, embedding, ntm_mask, 'n');
    let threat_subnet = operations::add(builder, stm_threat_subnet, ntm_threat_subnet);

    let output_bias = builder.create_weights("ob", Shape::new(3, 1));
    let dot_prod = operations::submatrix_product(builder, OUT_DIM, main_subnet, threat_subnet);
    let predicted = operations::add(builder, dot_prod, output_bias);
    operations::softmax_crossentropy_loss(builder, predicted, target);

    let ctx = ExecutionContext::default();
    builder.build(ctx)
}

fn embedding(builder: &mut GraphBuilder, inputs: Node) -> Node {
    let l0w = builder.create_weights("embw", Shape::new(SIZE, inputs::INPUT_SIZE));
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
    let l1w = builder.create_weights(&format!("{side}1w"), Shape::new(threats::NUM_THREATS, SIZE / 2));
    let l1b = builder.create_weights(&format!("{side}1b"), Shape::new(threats::NUM_THREATS, 1));
    let l2w = builder.create_weights(&format!("{side}2w"), Shape::new(OUT_DIM, threats::NUM_THREATS));
    let l2b = builder.create_weights(&format!("{side}2b"), Shape::new(OUT_DIM, 1));

    let out = operations::affine(builder, l1w, inputs, l1b);
    let out = operations::activate(builder, out, Activation::SCReLU);
    let out = operations::mask(builder, out, masks);
    operations::affine(builder, l2w, out, l2b)
}
