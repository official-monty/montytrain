use bullet::{
    operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

use crate::input::ThreatInputs;

pub fn make_trainer(filters: usize, l1: usize, l2: usize, l3: usize) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let (mut graph, output_node) = build_network(filters, l1, l2, l3);

    let mut save = Vec::new();

    let stdev = 1.0 / 3072f32.sqrt();
    graph.get_weights_mut("l0w").seed_random(0.0, stdev, true);
    save.push(("l0w".to_string(), QuantTarget::Float));

    let stdev = 1.0 / ((64 * filters) as f32).sqrt();
    graph.get_weights_mut("l1w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l1b").seed_random(0.0, stdev, true);
    save.push(("l1w".to_string(), QuantTarget::Float));
    save.push(("l1b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / (l1 as f32).sqrt();
    graph.get_weights_mut("l2w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l2b").seed_random(0.0, stdev, true);
    save.push(("l2w".to_string(), QuantTarget::Float));
    save.push(("l2b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / (l1 as f32).sqrt();
    graph.get_weights_mut("l2w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l2b").seed_random(0.0, stdev, true);
    save.push(("l2w".to_string(), QuantTarget::Float));
    save.push(("l2b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / (l2 as f32).sqrt();
    graph.get_weights_mut("l3w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l3b").seed_random(0.0, stdev, true);
    save.push(("l3w".to_string(), QuantTarget::Float));
    save.push(("l3b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / (l3 as f32).sqrt();
    graph.get_weights_mut("l4w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l4b").seed_random(0.0, stdev, true);
    save.push(("l4w".to_string(), QuantTarget::Float));
    save.push(("l4b".to_string(), QuantTarget::Float));

    Trainer::new(graph, output_node, AdamWParams::default(), ThreatInputs, outputs::Single, save, true)
}

fn convolution(builder: &mut GraphBuilder, input: Node, channels: (usize, usize), id: &str) -> Node {
    let w = builder.create_weights(&format!("{id}w"), Shape::new(9, channels.0 * channels.1));

    let conv_desc = ConvolutionDescription::new(
        Shape::new(8, 8),
        channels.0,
        channels.1,
        Shape::new(3, 3),
        Shape::new(1, 1),
        Shape::new(1, 1),
    );

    operations::convolution(builder, w, input, conv_desc)
}

fn build_network(filters: usize, l1: usize, l2: usize, l3: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(3072, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let mut out = convolution(&mut builder, stm, (48, filters), "l0");

    let l1w = builder.create_weights("l1w", Shape::new(l1, 64 * filters));
    let l1b = builder.create_weights("l1b", Shape::new(l1, 1));
    out = operations::affine(&mut builder, l1w, out, l1b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);

    let l2w = builder.create_weights("l2w", Shape::new(l2, l1));
    let l2b = builder.create_weights("l2b", Shape::new(l2, 1));
    out = operations::affine(&mut builder, l2w, out, l2b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);

    let l3w = builder.create_weights("l3w", Shape::new(l3, l2));
    let l3b = builder.create_weights("l3b", Shape::new(l3, 1));
    out = operations::affine(&mut builder, l3w, out, l3b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);

    let l4w = builder.create_weights("l4w", Shape::new(3, l3));
    let l4b = builder.create_weights("l4b", Shape::new(3, 1));
    out = operations::affine(&mut builder, l4w, out, l4b);

    operations::softmax_crossentropy_loss(&mut builder, out, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), out)
}
