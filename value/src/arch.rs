use crate::piecethreatbuckets::PieceThreatCount;
use bullet::{
    game::inputs::SparseInputType,
    nn::{
        optimiser::{AdamW, AdamWOptimiser},
        InitSettings, Shape,
    },
    trainer::save::SavedFormat,
    value::{ValueTrainer, ValueTrainerBuilder},
};

pub fn make_trainer<T: Default + SparseInputType>(
    l1: usize,
) -> ValueTrainer<AdamWOptimiser, T, PieceThreatCount> {
    let inputs = T::default();
    let num_inputs = inputs.num_inputs();

    ValueTrainerBuilder::default()
        .wdl_output()
        .inputs(T::default())
        .output_buckets(PieceThreatCount)
        .optimiser(AdamW)
        .save_format(&[
            SavedFormat::id("pst"),
            SavedFormat::id("l0w").quantise::<i16>(512),
            SavedFormat::id("l0b").quantise::<i16>(512),
            SavedFormat::id("l1w").quantise::<i16>(1024).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(1024),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build_custom(|builder, inputs, targets, output_buckets| {
            let num_buckets = PieceThreatCount::BUCKETS;
            let pst = builder.new_weights("pst", Shape::new(3, num_inputs), InitSettings::Zeroed);
            let l0 = builder.new_affine("l0", num_inputs, l1);
            let l1 = builder.new_affine("l1", l1 / 2, num_buckets * 16);
            let l2 = builder.new_affine("l2", 16, num_buckets * 128);
            let l3 = builder.new_affine("l3", 128, num_buckets * 3);

            l0.init_with_effective_input_size(128);

            let l0 = l0.forward(inputs).crelu().pairwise_mul();
            let l1 = l1.forward(l0).select(output_buckets).screlu();
            let l2 = l2.forward(l1).select(output_buckets).screlu();
            let l3 = l3.forward(l2).select(output_buckets);
            let out = l3 + pst.matmul(inputs);

            let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
            let loss = ones.matmul(out.softmax_crossentropy_loss(targets));

            (out, loss)
        })
}
