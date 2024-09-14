use montytrain::train;

fn main() {
    let mut args = std::env::args();
    args.next();
    let buffer_size_mb = args.next().unwrap().parse().unwrap();

    train(
        buffer_size_mb,
        "data/policygen6.binpack".to_string(),
        32,
        0.001,
        0.000001,
        32,
    ).unwrap();
}
