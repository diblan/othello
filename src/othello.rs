use crate::{board::Board, board_math::BoardMath, game::Game};

#[derive(Clone)]
pub struct Othello {
    n: usize,
    b: Board,
    bm: BoardMath,
}

impl Othello {
    const SQUARE_CONTENT: [(i8, &'static str); 3] = [(-1, "X"), (0, "-"), (1, "O")];
    pub fn display(board: &Vec<i8>, n: usize) {
        print!("   ");
        for y in 0..n {
            print!("{:?} ", y);
        }
        println!("");
        println!("--------------------");
        for x in 0..n {
            print!("{:?}| ", x);
            for y in 0..n {
                let mut content = "";
                for tup in Self::SQUARE_CONTENT {
                    if tup.0 == board[x * n + y] {
                        content = tup.1;
                    }
                }
                print!("{} ", content);
            }
            println!("|");
        }
        println!("--------------------");
    }
}

impl Game for Othello {
    fn new(n: usize) -> Self {
        Othello {
            n,
            b: Board::new(n),
            bm: BoardMath::new(n),
        }
    }

    fn get_init_board(&self) -> &Vec<i8> {
        &self.b.pieces
    }

    fn get_board_size(&self) -> (i8, i8) {
        (self.n as i8, self.n as i8)
    }

    fn get_action_size(&self) -> usize {
        self.n * self.n + 1
    }

    fn get_next_state(&self, board: &Vec<i8>, player: i8, action: u8) -> (Vec<i8>, i8) {
        if action as usize == self.n * self.n {
            return (board.clone(), -player);
        }
        let mut b = Board::new(self.n);
        b.pieces = board.clone();
        b.execute_move(action, player);
        (b.pieces, -player)
    }

    fn get_valid_moves(&self, board: &Vec<i8>, player: i8) -> Vec<u8> {
        let mut valids = vec![0; self.get_action_size()];
        let mut b = Board::new(self.n);
        b.pieces = board.clone();
        let legal_moves = b.get_legal_moves(player);
        if legal_moves.len() == 0 {
            valids[self.get_action_size() - 1] = 1;
            return valids;
        }
        for legal_move in legal_moves {
            valids[legal_move as usize] = 1;
        }
        valids
    }

    /// return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
    /// player = 1
    fn get_game_ended(&self, board: &Vec<i8>, player: i8) -> i8 {
        let mut b = Board::new(self.n);
        b.pieces = board.clone();
        if b.has_legal_moves(player) {
            return 0;
        }
        if b.has_legal_moves(-player) {
            return 0;
        }
        if b.count_diff(player) > 0 {
            return 1;
        }
        -1
    }

    fn get_canonical_form(&self, board: &Vec<i8>, player: i8) -> Vec<i8> {
        let mut canon = board.clone();
        for x in 0..canon.len() {
            canon[x] *= player;
        }
        canon
    }

    fn get_symmetries(&self, board: &Vec<i8>, pi: &Vec<f32>) -> Vec<(Vec<i8>, Vec<f32>)> {
        vec![
            (board.clone(), pi.clone()),
            (
                BoardMath::apply_1d_i8(board, &self.bm.mirror_1d),
                BoardMath::apply_1d(pi, &self.bm.mirror_1d),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.l90_1d),
                BoardMath::apply_1d(pi, &self.bm.l90_1d),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.l90_1d_mirror),
                BoardMath::apply_1d(pi, &self.bm.l90_1d_mirror),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.r90_1d),
                BoardMath::apply_1d(pi, &self.bm.r90_1d),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.r90_1d_mirror),
                BoardMath::apply_1d(pi, &self.bm.r90_1d_mirror),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.l180_1d),
                BoardMath::apply_1d(pi, &self.bm.l180_1d),
            ),
            (
                BoardMath::apply_1d_i8(board, &self.bm.l180_1d_mirror),
                BoardMath::apply_1d(pi, &self.bm.l180_1d_mirror),
            ),
        ]
    }

    fn string_representation(&self, board: &Vec<i8>) -> String {
        let mut sr = String::with_capacity(board.len() * board.len());
        for x in 0..self.n {
            for y in 0..self.n {
                let mut content = "";
                for tup in Self::SQUARE_CONTENT {
                    if tup.0 == board[x * self.n + y] {
                        content = tup.1;
                    }
                }
                sr.push_str(content);
            }
        }
        sr
    }
}

#[cfg(test)]
mod tests;
