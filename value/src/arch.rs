use bullet::{
    game::inputs::SparseInputType,
    nn::{
        optimiser::{AdamW, AdamWOptimiser},
        InitSettings, Shape,
    },
    trainer::save::SavedFormat,
    value::{NoOutputBuckets, ValueTrainer, ValueTrainerBuilder},
};

use crate::input::{NUM_KING_BUCKETS, TOTAL_THREATS};

const FACTORISED_PIECE_LEN: usize = 768;
const PIECE_LEN: usize = FACTORISED_PIECE_LEN * NUM_KING_BUCKETS;
const L1: usize = 3072;

fn merge_factoriser(mut weights: Vec<f32>, factoriser: &[f32], output_size: usize) -> Vec<f32> {
    let expanded = factoriser.repeat(NUM_KING_BUCKETS);
    let input_size = weights.len() / output_size;
    for row in 0..output_size {
        let w_start = row * input_size + TOTAL_THREATS;
        let w_end = w_start + PIECE_LEN;
        let f_start = row * PIECE_LEN;
        let f_end = f_start + PIECE_LEN;
        for (w, &f) in weights[w_start..w_end]
            .iter_mut()
            .zip(expanded[f_start..f_end].iter())
        {
            *w += f;
        }
    }
    weights
}

pub fn make_trainer<T: Default + SparseInputType>(
    l1: usize,
) -> ValueTrainer<AdamWOptimiser, T, NoOutputBuckets> {
    let inputs = T::default();
    let num_inputs = inputs.num_inputs();

    ValueTrainerBuilder::default()
        .wdl_output()
        .inputs(T::default())
        .optimiser(AdamW)
        .save_format(&[
            SavedFormat::id("pst").add_transform(|_, _, mut weights| {
                let factoriser = weights.split_off(weights.len() - 3 * FACTORISED_PIECE_LEN);
                merge_factoriser(weights, &factoriser, 3)
            }),
            SavedFormat::id("l0w")
                .add_transform({
                    move |_, _, mut weights| {
                        let factoriser =
                            weights.split_off(weights.len() - L1 * FACTORISED_PIECE_LEN);
                        merge_factoriser(weights, &factoriser, L1)
                    }
                })
                .quantise::<i16>(512),
            SavedFormat::id("l0b").quantise::<i16>(512),
            SavedFormat::id("l1w").quantise::<i16>(1024).transpose(),
            SavedFormat::id("l1b").quantise::<i16>(1024),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build_custom(|builder, inputs, targets| {
            let pst = builder.new_weights("pst", Shape::new(3, num_inputs), InitSettings::Zeroed);
            let l0 = builder.new_affine("l0", num_inputs, l1);
            let l1 = builder.new_affine("l1", l1 / 2, 16);
            let l2 = builder.new_affine("l2", 16, 128);
            let l3 = builder.new_affine("l3", 128, 3);

            l0.init_with_effective_input_size(128);

            let l0 = l0.forward(inputs).crelu().pairwise_mul();
            let l1 = l1.forward(l0).screlu();
            let l2 = l2.forward(l1).screlu();
            let l3 = l3.forward(l2);
            let out = l3 + pst.matmul(inputs);

            let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
            let loss = ones.matmul(out.softmax_crossentropy_loss(targets));

            (out, loss)
        })
}
