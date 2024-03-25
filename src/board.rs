use std::{collections::HashSet, vec};

#[derive(Clone)]
pub struct Board {
    n: usize,
    pub pieces: Vec<Vec<i8>>,
}

impl Board {
    pub const DIRECTIONS: [(i8, i8); 8] = [
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    ];

    pub fn new(n: usize) -> Self {
        let mut b = Board {
            n,
            pieces: vec![vec![0; n]; n],
        };
        b.pieces[n / 2 - 1][n / 2] = 1;
        b.pieces[n / 2][n / 2 - 1] = 1;
        b.pieces[n / 2 - 1][n / 2 - 1] = -1;
        b.pieces[n / 2][n / 2] = -1;
        b
    }

    pub fn count_diff(&self, color: i8) -> i32 {
        let mut count: i32 = 0;
        for row in &self.pieces {
            for square in row {
                if *square == color {
                    count += 1;
                }
                if *square == -color {
                    count -= 1;
                }
            }
        }
        count
    }

    pub fn get_legal_moves(&self, color: i8) -> Vec<(i8, i8)> {
        let mut moves: HashSet<(i8, i8)> = HashSet::new();
        for y in 0..self.n {
            for x in 0..self.n {
                if self.pieces[x][y] == color {
                    let new_moves = self.get_moves_for_square((x as i8, y as i8));
                    for new_move in new_moves {
                        moves.insert(new_move);
                    }
                }
            }
        }
        Vec::from_iter(moves)
    }

    pub fn has_legal_moves(&self, color: i8) -> bool {
        for y in 0..self.n {
            for x in 0..self.n {
                if self.pieces[x][y] == color {
                    let new_moves = self.get_moves_for_square((x as i8, y as i8));
                    if new_moves.len() > 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn get_moves_for_square(&self, square: (i8, i8)) -> Vec<(i8, i8)> {
        let color = self.pieces[square.0 as usize][square.1 as usize];

        if color == 0 {
            return vec![];
        }
        let mut moves: Vec<(i8, i8)> = vec![];
        for direction in Self::DIRECTIONS {
            let action = self.discover_move(square, direction);
            match action {
                Some(x) => moves.push(x),
                None => continue,
            }
        }
        moves
    }

    pub fn execute_move(&mut self, action: (i8, i8), color: i8) {
        for direction in Self::DIRECTIONS {
            for flip in self.get_flips(action, direction, color) {
                self.pieces[flip.0 as usize][flip.1 as usize] = color;
            }
        }
    }

    pub fn discover_move(&self, origin: (i8, i8), direction: (i8, i8)) -> Option<(i8, i8)> {
        let color = self.pieces[origin.0 as usize][origin.1 as usize];
        let mut flips: Vec<(i8, i8)> = vec![];

        for (x, y) in self.increment_move(origin, direction, self.n) {
            if self.pieces[x as usize][y as usize] == 0 {
                if flips.len() > 0 {
                    return Some((x, y));
                }
                return None;
            }
            if self.pieces[x as usize][y as usize] == color {
                return None;
            }
            if self.pieces[x as usize][y as usize] == -color {
                flips.push((x, y));
            }
        }
        None
    }

    pub fn get_flips(&self, origin: (i8, i8), direction: (i8, i8), color: i8) -> Vec<(i8, i8)> {
        let mut flips: Vec<(i8, i8)> = vec![origin];
        for dir in self.increment_move(origin, direction, self.n) {
            if self.pieces[dir.0 as usize][dir.1 as usize] == 0 {
                return vec![];
            }
            if self.pieces[dir.0 as usize][dir.1 as usize] == -color {
                flips.push(dir);
            } else {
                return flips;
            }
        }
        vec![]
    }

    pub fn increment_move(&self, action: (i8, i8), direction: (i8, i8), n: usize) -> Vec<(i8, i8)> {
        let mut output: Vec<(i8, i8)> = Vec::new();
        let mut x: i8 = action.0 + direction.0;
        let mut y: i8 = action.1 + direction.1;
        while x >= 0 && x < n as i8 && y >= 0 && y < n as i8 {
            output.push((x.clone(), y.clone()));
            x += direction.0;
            y += direction.1;
        }
        output
    }

    // pub fn get_item(&self, i: usize) -> &Vec<i8> {
    //     &self.pieces[i]
    // }

    // pub fn print_board(&self) {
    //     println!("board:");
    //     for row in &self.pieces {
    //         println!("{:?}", row);
    //     }
    // }

    // pub fn print_vec_board(board: &Vec<Vec<i8>>) {
    //     println!("board:");
    //     for row in board {
    //         println!("{:?}", row);
    //     }
    // }
}
