use crate::game::Game;

pub struct Arena<'a, F, F2, G, D>
where
    F: FnMut(&Vec<i8>) -> usize,
    F2: FnMut(&Vec<i8>) -> usize,
    G: Game,
    D: Fn(&Vec<i8>, usize),
{
    player1: F,
    player2: F2,
    game: &'a G,
    display: D,
}

/// An Arena class where any 2 agents can be pit against each other.
impl<'a, F, F2, G, D> Arena<'a, F, F2, G, D>
where
    F: FnMut(&Vec<i8>) -> usize,
    F2: FnMut(&Vec<i8>) -> usize,
    G: Game,
    D: Fn(&Vec<i8>, usize),
{
    /// Input:
    ///     player 1,2: two functions that takes board as input, return action
    ///     game: Game object
    ///     display: a function that takes board as input and prints it (e.g.
    ///              display in othello/OthelloGame). Is necessary for verbose
    ///              mode.
    ///
    /// see othello/OthelloPlayers.py for an example. See pit.py for pitting
    /// human players/other baselines with each other.
    pub fn new(player1: F, player2: F2, game: &'a G, display: D) -> Self {
        Arena {
            player1,
            player2,
            game,
            display,
        }
    }

    /// Executes one episode of a game.
    ///
    /// Returns:
    /// either
    ///     winner: player who won the game (1 if player1, -1 if player2)
    /// or
    ///     draw result returned from the game that is neither 1, -1, nor 0.
    pub fn play_game(&mut self, start_player: i8, verbose: bool) -> i8 {
        let mut cur_player = start_player;
        let mut board = self.game.get_init_board().clone();
        let mut it = 0;
        while self.game.get_game_ended(&board, cur_player) == 0 {
            it += 1;
            if verbose {
                println!("Turn {:?} Player {:?}", it, cur_player);
                (self.display)(&board, self.game.get_board_size().0 as usize);
            }
            let canon = self.game.get_canonical_form(&board, cur_player);
            let action = if cur_player == 1 {
                (self.player1)(&canon)
            } else {
                (self.player2)(&canon)
            };

            let valids = self.game.get_valid_moves(&canon, 1);

            if valids[action] == 0 {
                println!("Action {action} is not valid");
                println!("valids = {:?}", valids);
                assert!(valids[action] > 0);
            }
            let next_state = self.game.get_next_state(&board, cur_player, action as u8);
            board = next_state.0;
            cur_player = next_state.1;
        }
        if verbose {
            println!(
                "Game over: Turn {it} Result {:?}",
                self.game.get_game_ended(&board, 1)
            );
            (self.display)(&board, self.game.get_board_size().0 as usize);
        }
        return cur_player * self.game.get_game_ended(&board, cur_player);
    }

    /// Plays num games in which player1 starts num/2 games and player2 starts
    /// num/2 games.
    ///
    /// Returns:
    ///     oneWon: games won by player1
    ///     twoWon: games won by player2
    ///     draws:  games won by nobody
    pub fn play_games(&mut self, num: usize, verbose: bool) -> (i32, i32, i32) {
        let num = num / 2;
        let mut one_won = 0;
        let mut two_won = 0;
        let mut draws = 0;
        for _ in 0..num {
            let game_result = self.play_game(1, verbose);
            if game_result == 1 {
                one_won += 1;
            } else if game_result == -1 {
                two_won += 1;
            } else {
                draws += 1;
            }
        }
        for _ in 0..num {
            let game_result = self.play_game(-1, verbose);
            if game_result == -1 {
                one_won += 1;
            } else if game_result == 1 {
                two_won += 1;
            } else {
                draws += 1;
            }
        }
        return (one_won, two_won, draws);
    }
}
