use bullet::{
    operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

use crate::input::ThreatInputs;

pub fn make_trainer(blocks: usize, filters: usize, subsequent: &[usize]) -> Trainer<AdamWOptimiser, ThreatInputs, outputs::Single> {
    let (mut graph, output_node) = build_network(blocks, filters, subsequent);

    let mut save = Vec::new();

    let stdev = 1.0 / 3072f32.sqrt();
    graph.get_weights_mut("iw").seed_random(0.0, stdev, true);
    graph.get_weights_mut("ib").seed_random(0.0, stdev, true);
    save.push(("iw".to_string(), QuantTarget::Float));
    save.push(("ib".to_string(), QuantTarget::Float));

    let stdev = 1.0 / ((64 * filters) as f32).sqrt();

    for block in 0..blocks {
        graph.get_weights_mut(&format!("r{block}c1w")).seed_random(0.0, stdev, true);
        graph.get_weights_mut(&format!("r{block}c1b")).seed_random(0.0, stdev, true);
        save.push((format!("r{block}c1w"), QuantTarget::Float));
        save.push((format!("r{block}c1b"), QuantTarget::Float));

        graph.get_weights_mut(&format!("r{block}c2w")).seed_random(0.0, stdev, true);
        graph.get_weights_mut(&format!("r{block}c2b")).seed_random(0.0, stdev, true);
        save.push((format!("r{block}c2w"), QuantTarget::Float));
        save.push((format!("r{block}c2b"), QuantTarget::Float));
    }

    let mut inputs = 64 * filters;

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

    Trainer::new(graph, output_node, AdamWParams::default(), ThreatInputs, outputs::Single, save, true)
}

fn convolution(builder: &mut GraphBuilder, input: Node, channels: (usize, usize), id: &str) -> Node {
    let w = builder.create_weights(&format!("{id}w"), Shape::new(9, channels.0 * channels.1));
    let b = builder.create_weights(&format!("{id}b"), Shape::new(64 * channels.1, 1));

    let conv_desc = ConvolutionDescription::new(
        Shape::new(8, 8),
        channels.0,
        channels.1,
        Shape::new(3, 3),
        Shape::new(1, 1),
        Shape::new(1, 1),
    );

    let out = operations::convolution(builder, w, input, conv_desc);

    operations::add(builder, out, b)
}

fn build_network(blocks: usize, filters: usize, subsequent: &[usize]) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(3072, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let mut out = convolution(&mut builder, stm, (48, filters), "i");

    for block in 0..blocks {
        let conv1 = convolution(&mut builder, out, (filters, filters), &format!("r{block}c1"));
        let conv2 = convolution(&mut builder, conv1, (filters, filters), &format!("r{block}c2"));
        out = operations::add(&mut builder, out, conv2);
        out = operations::activate(&mut builder, out, Activation::ReLU);
    }

    let mut inputs = 64 * filters;

    for (i, &size) in subsequent.iter().enumerate() {
        let l1w = builder.create_weights(&format!("l{i}w"), Shape::new(size, inputs));
        let l1b = builder.create_weights(&format!("l{i}b"), Shape::new(size, 1));
        out = operations::affine(&mut builder, l1w, out, l1b);
        out = operations::activate(&mut builder, out, Activation::SCReLU);

        inputs = size;
    }

    let l4w = builder.create_weights("ow", Shape::new(3, inputs));
    let l4b = builder.create_weights("ob", Shape::new(3, 1));
    out = operations::affine(&mut builder, l4w, out, l4b);

    operations::softmax_crossentropy_loss(&mut builder, out, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), out)
}
