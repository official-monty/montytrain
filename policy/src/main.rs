mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    logger, lr, operations, optimiser::{AdamWOptimiser, AdamWParams, Optimiser}, save::{Layout, SavedFormat}, wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, QuantTarget, Shape, TrainingSchedule, TrainingSteps
};
use trainer::Trainer;

const ID: &str = "policy001";

fn main() {
    let data_preparer = preparer::DataPreparer::new("/home/privateclient/monty_value_training/interleaved.binpack", 96000);
    //let data_preparer = preparer::DataPreparer::new("data/policygen6.binpack", 4096);

    let size = 4096;
    let key_size = 32;

    let mut graph = network(size, key_size);

    unsafe {
        graph.get_input_mut("indices").load_sparse_from_slice(Shape::new(moves::NUM_MOVES, 1), moves::NUM_MOVES, &moves::indices());
        graph.get_input_mut("promos").load_sparse_from_slice(Shape::new(moves::NUM_MOVES, 1), moves::NUM_MOVES, &moves::promos());
    }

    graph.get_weights_mut("embw").seed_random(0.0, 1.0 / (inputs::INPUT_SIZE as f32).sqrt(), true);

    let stdev = 1.0 / ((size / 2) as f32).sqrt();
    graph.get_weights_mut("srcw").seed_random(0.0, stdev, true);
    graph.get_weights_mut("dstw").seed_random(0.0, stdev, true);
    graph.get_weights_mut("promow").seed_random(0.0, stdev, true);

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    let mut trainer = Trainer {
        optimiser: AdamWOptimiser::new(graph, optimiser_params),
    };

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
                            SavedFormat::new("embw", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("embb", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("srcw", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("srcb", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("dstw", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("dstb", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("promow", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("promob", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize, key_size: usize) -> Graph {
    let builder = &mut GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let mask = builder.create_input("mask", Shape::new(moves::NUM_MOVES, 1));
    let dist = builder.create_input("dist", Shape::new(moves::MAX_MOVES, 1));
    let indices = builder.create_input("indices", Shape::new(moves::NUM_MOVES, 1));
    let promos = builder.create_input("promos", Shape::new(moves::NUM_MOVES, 1));

    let embw = builder.create_weights("embw", Shape::new(size, inputs::INPUT_SIZE));
    let embb = builder.create_weights("embb", Shape::new(size, 1));
    let srcw = builder.create_weights("srcw", Shape::new(key_size * 64, size / 2));
    let srcb = builder.create_weights("srcb", Shape::new(key_size * 64, 1));
    let dstw = builder.create_weights("dstw", Shape::new(key_size * 128, size / 2));
    let dstb = builder.create_weights("dstb", Shape::new(key_size * 128, 1));
    let promow = builder.create_weights("promow", Shape::new(8, size / 2));
    let promob = builder.create_weights("promob", Shape::new(8, 1));

    let embed = operations::affine(builder, embw, inputs, embb);
    let embed = operations::activate(builder, embed, Activation::CReLU);
    let embed = operations::pairwise_mul(builder, embed);
    
    let src = operations::affine(builder, srcw, embed, srcb);
    let dst = operations::affine(builder, dstw, embed, dstb);
    let promo_subnet = operations::affine(builder, promow, embed, promob);

    let src_dst_logits = operations::submatrix_product(builder, key_size, src, dst);
    let src_dst_logits = operations::gather(builder, src_dst_logits, indices);
    let promo_logits = operations::gather(builder, promo_subnet, promos);
    let logits = operations::add(builder, src_dst_logits, promo_logits);

    operations::sparse_softmax_crossentropy_loss_masked(builder, mask, logits, dist);

    let ctx = ExecutionContext::default();
    builder.build(ctx)
}
