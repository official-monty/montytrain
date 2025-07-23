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

pub fn make_trainer<T: Default + SparseInputType>(l1: usize) -> Trainer<AdamWOptimiser, T, outputs::Single> {
    let inputs = T::default();
    let num_inputs = inputs.num_inputs();
    let nnz = inputs.max_active();

    let (mut graph, output_node) = build_network(num_inputs, nnz, l1);

    let sizes = [num_inputs, l1 / 2, 16, 128];

    // seed biases because huge input featureset can be weird
    for (i, &size) in sizes.iter().enumerate() {
        graph.get_weights_mut(&format!("l{i}b")).seed_random(0.0, 1.0 / (size as f32).sqrt(), true).unwrap();
    }

    Trainer::new(
        graph,
        output_node,
        AdamWParams::default(),
        inputs,
        outputs::Single,
        vec![
            SavedFormat::new("pst", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l0w", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l0b", QuantTarget::I16(512), Layout::Normal),
            SavedFormat::new("l1w", QuantTarget::I16(1024), Layout::Transposed(Shape::new(16, l1 / 2))),
            SavedFormat::new("l1b", QuantTarget::I16(1024), Layout::Normal),
            SavedFormat::new("l2w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l2b", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3w", QuantTarget::Float, Layout::Normal),
            SavedFormat::new("l3b", QuantTarget::Float, Layout::Normal),
        ],
        false,
    )
}

fn build_network(inputs: usize, nnz: usize, l1: usize) -> (Graph, Node) {
    let builder = NetworkBuilder::default();

    // inputs
    let stm = builder.new_sparse_input("stm", Shape::new(inputs, 1), nnz);
    let targets = builder.new_dense_input("targets", Shape::new(3, 1));

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
    out = out + pst.matmul(stm);
    out.softmax_crossentropy_loss(targets);

    // graph, output node
    let output_node = out.node();
    (builder.build(ExecutionContext::default()), output_node)
}
