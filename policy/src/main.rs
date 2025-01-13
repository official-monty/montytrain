mod inputs;
mod loader;
mod moves;
mod preparer;
mod trainer;

use bullet::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams, Optimiser},
        Activation, ExecutionContext, Graph, NetworkBuilder, Shape,
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
    let data_preparer = preparer::DataPreparer::new(
        "/home/privateclient/monty_value_training/interleaved.binpack",
        96000,
    );

    let size = 12288;

    let graph = network(size);

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
                            SavedFormat::new("l0w", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l0b", QuantTarget::Float, Layout::Normal),
                            SavedFormat::new("l1w", QuantTarget::Float, Layout::Transposed),
                            SavedFormat::new("l1b", QuantTarget::Float, Layout::Normal),
                        ],
                    )
                    .unwrap();
            }
        },
    );
}

fn network(size: usize) -> Graph {
    let builder = NetworkBuilder::default();

    let inputs = builder.new_input("inputs", Shape::new(inputs::INPUT_SIZE, 1));
    let mask = builder.new_input("mask", Shape::new(moves::NUM_MOVES, 1));
    let dist = builder.new_input("dist", Shape::new(moves::MAX_MOVES, 1));

    let l0 = builder.new_affine("l0", inputs::INPUT_SIZE, size);
    let l1 = builder.new_affine("l1", size / 2, moves::NUM_MOVES);

    let mut out = l0.forward(inputs).activate(Activation::CReLU);
    out = out.pairwise_mul();
    out = l1.forward(out);
    out.masked_softmax_crossentropy_loss(dist, mask);

    builder.build(ExecutionContext::default())
}
