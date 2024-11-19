use bullet::{
    inputs::InputType, operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

use crate::input::ThreatInputs;

pub fn make_trainer(input_channels: usize, output_channels: usize, subsequent: &[usize]) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let input_size = ThreatInputs.size();

    let (mut graph, output_node) = build_network(input_size, input_channels, output_channels, subsequent);

    let mut save = Vec::new();

    let stdev = 1.0 / (input_size as f32).sqrt();
    graph.get_weights_mut("i0w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("i0b").seed_random(0.0, stdev, true);
    save.push(("i0w".to_string(), QuantTarget::Float));
    save.push(("i0b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / ((64 * input_channels) as f32).sqrt();
    graph.get_weights_mut("i1w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("i1b").seed_random(0.0, stdev, true);
    save.push(("i1w".to_string(), QuantTarget::Float));
    save.push(("i1b".to_string(), QuantTarget::Float));

    let mut inputs = 16 * output_channels;

    for (i, &size) in subsequent.iter().enumerate() {
        let stdev = 1.0 / (inputs as f32).sqrt();
        graph.get_weights_mut(&format!("l{i}w")).seed_random(0.0, stdev, true);
        graph.get_weights_mut(&format!("l{i}b")).seed_random(0.0, stdev, true);
        save.push((format!("l{i}w"), QuantTarget::Float));
        save.push((format!("l{i}b"), QuantTarget::Float));

        inputs = size;
    }

    let stdev = 1.0 / (inputs as f32).sqrt();
    graph.get_weights_mut("ow").seed_random(0.0, stdev, true);
    graph.get_weights_mut("ob").seed_random(0.0, stdev, true);
    save.push(("ow".to_string(), QuantTarget::Float));
    save.push(("ob".to_string(), QuantTarget::Float));

    Trainer::new(graph, output_node, AdamWParams::default(), ThreatInputs, outputs::Single, save, false)
}

fn build_network(input_size: usize, input_channels: usize, output_channels: usize, subsequent: &[usize]) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(input_size, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let i0w = builder.create_weights("i0w", Shape::new(64 * input_channels, input_size));
    let i0b = builder.create_weights("i0b", Shape::new(64 * input_channels, 1));
    let i1w = builder.create_weights("i1w", Shape::new(25, input_channels * output_channels));
    let i1b = builder.create_weights("i1b", Shape::new(16 * output_channels, 1));

    let conv_desc = ConvolutionDescription::new(
        Shape::new(8, 8),
        input_channels,
        output_channels,
        Shape::new(5, 5),
        (0, 0),
        Shape::new(1, 1),
    );

    let mut out = operations::affine(&mut builder, i0w, stm, i0b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);
    out = operations::convolution(&mut builder, i1w, out, conv_desc);
    out = operations::add(&mut builder, out, i1b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);

    let mut inputs = 16 * output_channels;
    for (i, &size) in subsequent.iter().enumerate() {
        let l1w = builder.create_weights(&format!("l{i}w"), Shape::new(size, inputs));
        let l1b = builder.create_weights(&format!("l{i}b"), Shape::new(size, 1));
        out = operations::affine(&mut builder, l1w, out, l1b);
        out = operations::activate(&mut builder, out, Activation::SCReLU);
        inputs = size;
    }

    let ow = builder.create_weights("ow", Shape::new(3, inputs));
    let ob = builder.create_weights("ob", Shape::new(3, 1));
    out = operations::affine(&mut builder, ow, out, ob);

    operations::softmax_crossentropy_loss(&mut builder, out, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), out)
}
