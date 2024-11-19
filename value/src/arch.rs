use bullet::{
    inputs, operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

pub fn make_trainer(filters: usize, hl: usize) -> Trainer<AdamWOptimiser, inputs::Chess768, outputs::Single> {
    let (mut graph, output_node) = build_network(filters, hl);

    let mut save = Vec::new();

    let stdev = 1.0 / 768f32.sqrt();
    graph.get_weights_mut("l0w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l0b").seed_random(0.0, stdev, true);
    save.push(("l0w".to_string(), QuantTarget::Float));
    save.push(("l0b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / ((64 * filters) as f32).sqrt();
    graph.get_weights_mut("l1w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l1b").seed_random(0.0, stdev, true);
    save.push(("l1w".to_string(), QuantTarget::Float));
    save.push(("l1b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / (hl as f32).sqrt();
    graph.get_weights_mut("l2w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l2b").seed_random(0.0, stdev, true);
    save.push(("l2w".to_string(), QuantTarget::Float));
    save.push(("l2b".to_string(), QuantTarget::Float));

    Trainer::new(graph, output_node, AdamWParams::default(), inputs::Chess768, outputs::Single, save, true)
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

    let conv = operations::convolution(builder, w, input, conv_desc);

    operations::add(builder, conv, b)
}

fn build_network(filters: usize, hl: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let stm = builder.create_input("stm", Shape::new(768, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let mut out = convolution(&mut builder, stm, (12, filters), "l0");

    let l1w = builder.create_weights("l1w", Shape::new(hl, 64 * filters));
    let l1b = builder.create_weights("l1b", Shape::new(hl, 1));
    out = operations::affine(&mut builder, l1w, out, l1b);
    out = operations::activate(&mut builder, out, Activation::SCReLU);

    let l2w = builder.create_weights("l2w", Shape::new(3, hl));
    let l2b = builder.create_weights("l2b", Shape::new(3, 1));
    out = operations::affine(&mut builder, l2w, out, l2b);

    operations::softmax_crossentropy_loss(&mut builder, out, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), out)
}
