use bullet::{
    default::{Layout, SavedFormat}, inputs::SparseInputType, operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

use crate::input::ThreatInputs;

pub fn make_trainer(l1: usize) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let num_inputs = ThreatInputs.num_inputs();

    let (mut graph, output_node) = build_network(num_inputs, l1);

    let sizes = [num_inputs, l1, 16, 128];

    for (i, &size) in sizes.iter().enumerate() {
        graph
            .get_weights_mut(&format!("l{i}w"))
            .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);

        graph
            .get_weights_mut(&format!("l{i}b"))
            .seed_random(0.0, 1.0 / (size as f32).sqrt(), true);
    }

    Trainer::new(
        graph,
        output_node,
        AdamWParams::default(),
        ThreatInputs,
        outputs::Single,
        vec![
            SavedFormat::new("l0w", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(1024), Layout::Transposed),
            SavedFormat::new("l1b", QuantTarget::I16(1024), Layout::Transposed),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("pst", QuantTarget::Float, Layout::Normal),
        ],
        false,
    )
}

fn build_network(inputs: usize, l1: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(inputs, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    // trainable weights
    let pst = builder.create_weights("pstw", Shape::new(3, inputs));
    let l0w = builder.create_weights("l0w", Shape::new(l1, inputs));
    let l0b = builder.create_weights("l0b", Shape::new(l1, 1));
    let l1w = builder.create_weights("l1w", Shape::new(16, l1 / 2));
    let l1b = builder.create_weights("l1b", Shape::new(16, 1));
    let l2w = builder.create_weights("l2w", Shape::new(128, 16));
    let l2b = builder.create_weights("l2b", Shape::new(128, 1));
    let l3w = builder.create_weights("l3w", Shape::new(3, 128));
    let l3b = builder.create_weights("l3b", Shape::new(3, 1));

    // inference
    let l1 = operations::affine(&mut builder, l0w, stm, l0b);
    let l1 = operations::activate(&mut builder, l1, Activation::CReLU);
    let l1 = operations::pairwise_mul(&mut builder, l1);
    let l2 = operations::affine(&mut builder, l1w, l1, l1b);
    let l2 = operations::activate(&mut builder, l2, Activation::SCReLU);
    let l3 = operations::affine(&mut builder, l2w, l2, l2b);
    let l3 = operations::activate(&mut builder, l3, Activation::SCReLU);
    let l4 = operations::affine(&mut builder, l3w, l3, l3b);

    let pst = operations::matmul(&mut builder, pst, stm);

    let predicted = operations::add(&mut builder, l4, pst);
    operations::softmax_crossentropy_loss(&mut builder, predicted, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
