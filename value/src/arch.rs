use bullet::{
    nn::{
        optimiser::{AdamWOptimiser, AdamWParams},
        Activation, ExecutionContext, Graph, InitSettings, NetworkBuilder, Node, Shape,
    },
    trainer::{
        default::{inputs::SparseInputType, outputs, Trainer},
        save::{Layout, QuantTarget, SavedFormat},
    },
};

use crate::input::ThreatInputs;

pub fn make_trainer(l1: usize) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let num_inputs = ThreatInputs.num_inputs();

    let (graph, output_node) = build_network(num_inputs, l1);

    Trainer::new(
        graph,
        output_node,
        AdamWParams::default(),
        ThreatInputs,
        outputs::Single,
        vec![
            SavedFormat::new("pst", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l0w", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(1024), Layout::Transposed),
            SavedFormat::new("l1b", QuantTarget::I16(1024), Layout::Normal),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
        ],
        false,
    )
}

fn build_network(inputs: usize, l1: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_input("stm", Shape::new(inputs, 1));
    let targets = builder.new_input("targets", Shape::new(3, 1));

    // trainable weights
    let pst = builder.new_weights("pst", Shape::new(3, inputs), InitSettings::Zeroed);
    let l0 = builder.new_affine("l0", inputs, l1);
    let l1 = builder.new_affine("l1", l1 / 2, 16);
    let l2 = builder.new_affine("l2", 16, 128);
    let l3 = builder.new_affine("l3", 128, 3);

    // inference
    let mut out = l0.forward(stm).activate(Activation::CReLU);
    out = out.pairwise_mul();
    out = l1.forward(out).activate(Activation::SCReLU);
    out = l2.forward(out).activate(Activation::SCReLU);
    out = l3.forward(out);
    out += pst * stm;
    out.softmax_crossentropy_loss(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
