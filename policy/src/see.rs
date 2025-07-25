use montyformat::chess::{Castling, Move, Piece, Position};

pub fn get_resolved_see_pos(pos: &Position, castling: &Castling, mov: Move) -> Position {
    let mut board = *pos;
    board.make(mov, castling);

    see_rec(&board, mov.to(), castling).1
}

const SEE_VALS: [i32; 8] = [0, 0, 100, 450, 450, 650, 1250, 0];

fn see_rec(board: &Position, to: u16, castling: &Castling) -> (i32, Position) {
    let mut moves = Vec::new();
    board.map_legal_captures(castling, |mv| {
        if mv.to() == to || mv.is_en_passant() {
            moves.push(mv);
        }
    });

    if moves.is_empty() {
        return (0, *board);
    }

    let mut best = i32::MIN;
    let mut best_board = *board;

    for mv in moves {
        let captured = if mv.is_en_passant() { Piece::PAWN } else { board.get_pc(1 << to) };

        let mut tmp = *board;
        tmp.make(mv, castling);

        let mut gain = SEE_VALS[captured];
        if mv.is_promo() {
            gain += SEE_VALS[mv.promo_pc()] - SEE_VALS[Piece::PAWN];
        }

        let (rec_score, pos) = see_rec(&tmp, to, castling);

        let score = gain - rec_score;
        if score > best {
            best = score;
            best_board = pos;
        }
    }

    if best < 0 {
        (0, *board)
    } else {
        (best, best_board)
    }
}

pub fn test_see() {
    for (fen, mov, res) in [
        (
            "6k1/1pp4p/p1pb4/6q1/3P1pRr/2P4P/PP1Br1P1/5RKN w - - 0 1",
            "f1f4",
            "6k1/1pp4p/p1p5/6q1/3P1BRr/2P4P/PP2r1P1/6KN b - - 0 2",
        ),
        (
            "5rk1/1pp2q1p/p1pb4/8/3P1NP1/2P5/1P1BQ1P1/5RK1 b - - 0 1",
            "d6f4",
            "5rk1/1pp2q1p/p1p5/8/3P1BP1/2P5/1P2Q1P1/5RK1 b - - 0 2",
        ),
        (
            "4R3/2r3p1/5bk1/1p1r3p/p2PR1P1/P1BK1P2/1P6/8 b - - 0 1",
            "h5g4",
            "4R3/2r3p1/5bk1/1p1r4/p2PR1P1/P1BK4/1P6/8 b - - 0 2",
        ),
        ("3r3k/3r4/2n1n3/8/3p4/2PR4/1B1Q4/3R3K w - - 0 1", "d3d4", "7k/8/8/8/3R4/8/8/7K b - - 0 5"),
    ] {
        let mut castling = Castling::default();
        let position = Position::parse_fen(fen, &mut castling);

        let mut mv = None;
        position.map_legal_moves(&castling, |this| {
            if this.to_uci(&castling) == mov {
                mv = Some(this)
            }
        });

        let mv = mv.unwrap();

        let result = get_resolved_see_pos(&position, &castling, mv).as_fen();
        println!("FEN: {fen}");
        println!("Move: {mov}");
        println!("Result: {result}");

        assert_eq!(res, result);
    }
}
