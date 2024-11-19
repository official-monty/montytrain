use bullet::{
    inputs, operations, optimiser::{AdamWOptimiser, AdamWParams}, outputs, Activation, ConvolutionDescription, ExecutionContext, Graph, GraphBuilder, Node, QuantTarget, Shape, Trainer
};

pub fn make_trainer() -> Trainer<AdamWOptimiser, inputs::Chess768, outputs::Single> {
    let (mut graph, output_node) = build_network(16, 64);

    let mut save = Vec::new();

    let stdev = 1.0 / 768f32.sqrt();
    graph.get_weights_mut("l0f").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l0b").seed_random(0.0, stdev, true);
    save.push(("l0f".to_string(), QuantTarget::Float));
    save.push(("l0b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / 1024f32.sqrt();
    graph.get_weights_mut("l1w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l1b").seed_random(0.0, stdev, true);
    save.push(("l1w".to_string(), QuantTarget::Float));
    save.push(("l1b".to_string(), QuantTarget::Float));

    let stdev = 1.0 / 64f32.sqrt();
    graph.get_weights_mut("l2w").seed_random(0.0, stdev, true);
    graph.get_weights_mut("l2b").seed_random(0.0, stdev, true);
    save.push(("l2w".to_string(), QuantTarget::Float));
    save.push(("l2b".to_string(), QuantTarget::Float));

    Trainer::new(graph, output_node, AdamWParams::default(), inputs::Chess768, outputs::Single, save, true)
}

fn build_network(channels: usize, l2: usize) -> (Graph, Node) {
    let mut builder = GraphBuilder::default();

    // inputs
    let mut stm = builder.create_input("stm", Shape::new(768, 1));
    let targets = builder.create_input("targets", Shape::new(3, 1));

    let input_channels = 12;
    let filters = builder.create_weights("l0f", Shape::new(9, input_channels * channels));
    let bias = builder.create_weights("l0b", Shape::new(64 * channels, 1));
    let conv_desc = ConvolutionDescription::new(
        Shape::new(8, 8),
        input_channels,
        channels,
        Shape::new(3, 3),
        Shape::new(1, 1),
        Shape::new(1, 1),
    );

    stm = operations::convolution(&mut builder, filters, stm, conv_desc);
    stm = operations::add(&mut builder, stm, bias);
    stm = operations::activate(&mut builder, stm, Activation::ReLU);

    let l1w = builder.create_weights("l1w", Shape::new(l2, 64 * channels));
    let l1b = builder.create_weights("l1b", Shape::new(l2, 1));
    stm = operations::affine(&mut builder, l1w, stm, l1b);

    let l2w = builder.create_weights("l2w", Shape::new(3, l2));
    let l2b = builder.create_weights("l2b", Shape::new(3, 1));
    let predicted = operations::affine(&mut builder, l2w, stm, l2b);

    operations::softmax_crossentropy_loss(&mut builder, predicted, targets);

    // graph, output node
    (builder.build(ExecutionContext::default()), predicted)
}
