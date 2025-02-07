mod arch;
mod consts;
mod input;
mod threats;

use arch::make_trainer;

use bullet::trainer::{
    default::{
        formats::sfbinpack::{
            chess::{piecetype::PieceType, r#move::MoveType},
            TrainingDataEntry,
        },
        inputs::ChessBucketsMirrored,
        loader,
    },
    schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
    settings::LocalSettings,
};

const HIDDEN_SIZE: usize = 2048;

fn main() {
    #[rustfmt::skip]
    let inputs = ChessBucketsMirrored::new([
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31,
    ]);

    let mut trainer = make_trainer(inputs, HIDDEN_SIZE);

    // loading from a SF binpack
    let data_loader = {
        let file_path = "data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack";
        let buffer_size_mb = 8192;
        let threads = 4;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to).piece_type() == PieceType::None
                && !entry.pos.is_checked(entry.pos.side_to_move())
        }

        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    let schedule = TrainingSchedule {
        net_id: "4096EXP".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: 3000,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::ExponentialDecayLR {
            initial_lr: 0.001,
            final_lr: 0.0000001,
            final_superbatch: 3000,
        },
        save_rate: 100,
    };

    let settings = LocalSettings {
        threads: 8,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    trainer.run(&schedule, &settings, &data_loader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
