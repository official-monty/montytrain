use bullet::{
    nn::{
        optimiser::{AdamW, AdamWOptimiser},
        Activation,
    },
    trainer::default::{inputs::SparseInputType, outputs, Loss, Trainer, TrainerBuilder},
};

#[rustfmt::skip]
pub fn make_trainer<T: Default + SparseInputType>(
    inputs: T, l1: usize,
) -> Trainer<AdamWOptimiser, T, outputs::Single> {
    TrainerBuilder::default()
        .quantisations(&[255, 64])
        .optimiser(AdamW)
        .loss_fn(Loss::SigmoidMSE)
        .input(inputs)
        .output_buckets(outputs::Single)
        .feature_transformer(l1)
        .activate(Activation::SCReLU)
        .add_layer(1)
        .build()
}
