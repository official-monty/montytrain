mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    logger, lr, operations,
    optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
    wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Shape,
    TrainingSchedule, TrainingSteps,
};
use trainer::Trainer;

const ID: &str = "policy001";

fn main() {
    let data_preparer = preparer::DataPreparer::new("/home/privateclient/monty_value_training/interleaved-policy.binpack", 96000);

    let size = 4096;

    let mut graph = network(size);

    graph
        .get_weights_mut("l0w")
        .seed_random(0.0, 1.0 / (inputs::INPUT_SIZE as f32).sqrt(), true);
    graph
        .get_weights_mut("l1w")
        .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);

    graph
        .get_weights_mut("v1w")
        .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);
    graph
        .get_weights_mut("v2w")
        .seed_random(0.0, 1.0 / 16f32.sqrt(), true);
    graph
        .get_weights_mut("v3w")
        .seed_random(0.0, 1.0 / 128f32.sqrt(), true);

    let mut trainer = Trainer {
        optimiser: AdamWOptimiser::new(graph, AdamWParams::default()),
    };

    let schedule = TrainingSchedule {
        net_id: ID.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 300,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.00001,
            final_superbatch: 300,
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
                        &["l0w", "l0b", "l1w", "l1b", "v1w", "v1b", "v2w", "v2b", "v3w", "v3b", "pstw", "pstb"],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize) -> Graph {
    let mut builder = GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let mask = builder.create_input("mask", Shape::new(moves::NUM_MOVES, 1));
    let dist = builder.create_input("dist", Shape::new(moves::MAX_MOVES, 1));
    let wdl = builder.create_input("wdl", Shape::new(3, 1));

    let l0w = builder.create_weights("l0w", Shape::new(size, inputs::INPUT_SIZE));
    let l0b = builder.create_weights("l0b", Shape::new(size, 1));

    let l1w = builder.create_weights("l1w", Shape::new(moves::NUM_MOVES, size));
    let l1b = builder.create_weights("l1b", Shape::new(moves::NUM_MOVES, 1));

    let v1w = builder.create_weights("v1w", Shape::new(16, size));
    let v1b = builder.create_weights("v1b", Shape::new(16, 1));
    let v2w = builder.create_weights("v2w", Shape::new(128, 16));
    let v2b = builder.create_weights("v2b", Shape::new(128, 1));
    let v3w = builder.create_weights("v3w", Shape::new(3, 128));
    let v3b = builder.create_weights("v3b", Shape::new(3, 1));

    let pstw = builder.create_weights("pstw", Shape::new(3, inputs::INPUT_SIZE));
    let pstb = builder.create_weights("pstb", Shape::new(3, 1));

    let l1 = operations::affine(&mut builder, l0w, inputs, l0b);
    let backbone = operations::activate(&mut builder, l1, Activation::SCReLU);

    let policy = operations::affine(&mut builder, l1w, backbone, l1b);
    let policy_loss = operations::sparse_softmax_crossentropy_loss_masked(&mut builder, mask, policy, dist);

    let v2 = operations::affine(&mut builder, v1w, backbone, v1b);
    let v2 = operations::activate(&mut builder, v2, Activation::SCReLU);
    let v3 = operations::affine(&mut builder, v2w, v2, v2b);
    let v3 = operations::activate(&mut builder, v3, Activation::SCReLU);
    let v4 = operations::affine(&mut builder, v3w, v3, v3b);
    let pst = operations::affine(&mut builder, pstw, inputs, pstb);
    let value = operations::add(&mut builder, v4, pst);
    let value_loss = operations::softmax_crossentropy_loss(&mut builder, value, wdl);

    operations::add(&mut builder, policy_loss, value_loss);

    let ctx = ExecutionContext::default();
    builder.build(ctx)
}
