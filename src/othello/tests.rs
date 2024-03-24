use rand::{distributions::Uniform, Rng};

use super::Othello;
use crate::game::Game;

#[test]
fn get_init_board_4() {
    let othello = Othello::new(4);
    let init_board = othello.get_init_board();
    let init_board_str = format!("{:?}", init_board);
    let expected_board = "[0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0]";
    assert_eq!(init_board_str, expected_board);
}

#[test]
fn get_init_board_6() {
    let othello = Othello::new(6);
    let init_board = othello.get_init_board();
    let init_board_str = format!("{:?}", init_board);
    let expected_board = "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]";
    assert_eq!(init_board_str, expected_board);
}

#[test]
fn get_board_size_4() {
    let othello = Othello::new(4);
    let size = othello.get_board_size();
    assert_eq!(size, (4, 4));
}

#[test]
fn get_board_size_6() {
    let othello = Othello::new(6);
    let size = othello.get_board_size();
    assert_eq!(size, (6, 6));
}

#[test]
fn get_action_size_4() {
    let othello = Othello::new(4);
    let action_size = othello.get_action_size();
    assert_eq!(action_size, 17);
}

#[test]
fn get_action_size_6() {
    let othello = Othello::new(6);
    let action_size = othello.get_action_size();
    assert_eq!(action_size, 37);
}

#[test]
fn get_next_state_4() {
    let othello = Othello::new(4);
    let board = othello.get_init_board();
    let next_state = othello.get_next_state(board, 1, 1);
    let next_state_str = format!("{:?}", next_state);
    let expected_next_state = "([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0], -1)";
    assert_eq!(next_state_str, expected_next_state.to_string());

    let board = &next_state.0;
    let next_state = othello.get_next_state(board, -1, 0);
    let next_state_str = format!("{:?}", next_state);
    let expected_next_state = "([-1, 1, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0], 1)";
    assert_eq!(next_state_str, expected_next_state.to_string());
}

#[test]
fn get_next_state_illegal() {
    let othello = Othello::new(4);
    let board = othello.get_init_board();
    let next_state = othello.get_next_state(board, 1, 6);
    let next_state_str = format!("{:?}", next_state);
    let expected_next_state = "([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0], -1)";
    assert_eq!(next_state_str, expected_next_state.to_string());
}

#[test]
fn get_valid_moves_4() {
    let othello = Othello::new(4);
    let board = othello.get_init_board();
    let valid_moves = othello.get_valid_moves(board, 1);
    let valid_moves_str = format!("{:?}", valid_moves);
    let valid_moves_expected = "[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]";
    assert_eq!(valid_moves_str, valid_moves_expected);
}

#[test]
fn get_game_ended_4() {
    let othello = Othello::new(4);
    let board = [0, -1, -1, -1, -1, -1, 1, -1, 0, -1, -1, -1, -1, 0, -1, -1].to_vec();
    let game_ended = othello.get_game_ended(&board, 1);
    assert_eq!(game_ended, -1);
}

#[test]
fn mask_probability_vector_4() {
    let pi = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
    ];
    let othello = Othello::new(4);
    let board = othello.get_init_board();
    let valid_moves = othello.get_valid_moves(board, 1);
    let mut pi = pi
        .iter()
        .zip(valid_moves.iter())
        .map(|(x, y)| x * *y as f32)
        .collect::<Vec<f32>>();
    println!("{:?}", pi);
    let sum_ps_s = pi.iter().sum::<f32>();
    println!("sum_ps_s: {:?}", sum_ps_s);
    for i in pi.iter_mut() {
        *i /= sum_ps_s;
    }
    println!("{:?}", pi);
    let sum_ps_s = pi.iter().sum::<f32>();
    println!("sum_ps_s: {:?}", sum_ps_s);
}

#[test]
fn rand_index() {
    let rand_generator = rand::thread_rng();
    let distr = Uniform::new(0, 215);
    let sample_id_it: Vec<usize> = rand_generator.sample_iter(distr).take(64).collect();
    println!("{:?}", sample_id_it);
}

// fn board_test() {
//     // let g: Othello = Othello::new(8);
//     let mut b: Board = Board::new(8);
//     let t: i8 = b.pieces[3][4];
//     print!("{t}");
//     let r: &Vec<i8> = b.get_item(3);
//     let rr: i8 = r[4];
//     println!("{rr}");
//     let d = b.count_diff(1);
//     println!("{d}");
//     let u = b.increment_move((8, 0), (0, 1), 9);
//     println!("{:?}", u);
//     let f = b.get_flips((3, 2), (1, 1), 1);
//     print!("{:?}", f);
//     b.print_board();
//     b.execute_move((3, 2), 1);
//     b.print_board();
//     let discover = b.discover_move((4, 4), (0, -1));
//     println!("{:?}", discover);
// }

// fn othello_test() {
//     println!("othello_test");
//     let o: Othello = Othello::new(8);
//     let board: &Vec<Vec<i8>> = o.get_init_board();
//     let finished_board: Vec<Vec<i8>> = finished_board();
//     let player = 1;
//     println!("{:?}", board.len());
//     Board::print_vec_board(&board);
//     let bs = o.get_board_size().clone();
//     println!("{:?},{:?}", bs.0, bs.1);
//     println!("{:?}", o.get_action_size());
//     let ns = o.get_next_state(&board, 1, 34);
//     Board::print_vec_board(&ns.0);
//     println!("{:?}", &ns.1);
//     let vm = o.get_valid_moves(&board, player);
//     // match vm {
//     //     Some(valid_moves) => println!("{:?}", valid_moves),
//     //     None => print!("No valid moves."),
//     // }
//     println!("{:?}", o.get_game_ended(&board, player));
//     println!("{:?}", o.get_game_ended(&finished_board, player));

//     o.string_representation(&o.get_canonical_form(board, -player));

//     let pi = vec![0. as f32; o.get_action_size()];
//     let sym = o.get_symmetries(&board, &pi);
//     for (b, _p) in sym {
//         o.string_representation(&b);
//     }
// }

// fn finished_board() -> Vec<Vec<i8>> {
//     let n = 8;
//     let mut b = vec![vec![0; n]; n];
//     b[n / 2 - 1][n / 2] = 1;
//     b[n / 2][n / 2 - 1] = 1;
//     b[n / 2 - 1][n / 2 - 1] = 1;
//     b[n / 2][n / 2] = 1;
//     b
// }
