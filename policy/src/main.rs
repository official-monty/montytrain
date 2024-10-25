mod inputs;
mod loader;
mod moves;
mod trainer;
mod preparer;

use bullet::{lr, operations, optimiser::{AdamWOptimiser, AdamWParams, Optimiser}, wdl, Activation, ExecutionContext, Graph, GraphBuilder, LocalSettings, NetworkTrainer, Shape, TrainingSchedule, TrainingSteps};
use trainer::Trainer;

fn main() {
    println!("{}", moves::NUM_MOVES);
    let data_preparer = preparer::DataPreparer::new("../binpacks/policygen6", 4096);
    let graph = network(512);

    let mut trainer = Trainer {
        optimiser: AdamWOptimiser::new(graph, AdamWParams::default()),
    };

    let schedule = TrainingSchedule {
        net_id: "policy001".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 240,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 60 },
        save_rate: 150,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    trainer.train_custom(&data_preparer, &Option::<preparer::DataPreparer>::None, &schedule, &settings, |_, _, _, _| {});
}

fn network(size: usize) -> Graph {
    let mut builder = GraphBuilder::default();

    let inputs = builder.create_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let dist = builder.create_input("dist", Shape::new(moves::NUM_MOVES, 1));

    let l0w = builder.create_weights("l0w", Shape::new(size, inputs::INPUT_SIZE));
    let l0b = builder.create_weights("l0b", Shape::new(size, 1));

    let l1w = builder.create_weights("l1w", Shape::new(moves::NUM_MOVES, size));
    let l1b = builder.create_weights("l1b", Shape::new(moves::NUM_MOVES, 1));

    let l1 = operations::affine(&mut builder, l0w, inputs, l0b);
    let l1a = operations::activate(&mut builder, l1, Activation::SCReLU);
    let l2 = operations::affine(&mut builder, l1w, l1a, l1b);

    operations::softmax_crossentropy_loss(&mut builder, l2, dist);

    let ctx = ExecutionContext::default();
    builder.build(ctx)
}
