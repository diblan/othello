pub trait Game {
    /// This class specifies the base Game class. To define your own game, subclass
    /// this class and implement the functions below. This works when the game is
    /// two-player, adversarial and turn-based.
    ///
    /// Use 1 for player1 and -1 for player2.
    ///
    /// See othello/OthelloGame.py for an example implementation.
    fn new(n: usize) -> Self;

    /// Returns:
    ///     startBoard: a representation of the board (ideally this is the form
    ///                 that will be the input to your neural network)
    fn get_init_board(&self) -> &Vec<i8>;

    /// Returns:
    ///     (x,y): a tuple of board dimensions
    fn get_board_size(&self) -> (i8, i8);

    /// Returns:
    ///     actionSize: number of all possible actions
    fn get_action_size(&self) -> usize;

    /// Input:
    ///     board: current board
    ///     player: current player (1 or -1)
    ///     action: action taken by current player
    ///
    /// Returns:
    ///     nextBoard: board after applying action
    ///     nextPlayer: player who plays in the next turn (should be -player)
    fn get_next_state(&self, board: &Vec<i8>, player: i8, action: u8) -> (Vec<i8>, i8);

    /// Input:
    ///     board: current board
    ///     player: current player
    ///
    /// Returns:
    ///     validMoves: a binary vector of length self.getActionSize(), 1 for
    ///                 moves that are valid from the current board and player,
    ///                 0 for invalid moves
    fn get_valid_moves(&self, board: &Vec<i8>, player: i8) -> Vec<u8>;

    /// Input:
    ///     board: current board
    ///     player: current player (1 or -1)
    ///
    /// Returns:
    ///     r: 0 if game has not ended. 1 if player won, -1 if player lost,
    ///        small non-zero value for draw.
    fn get_game_ended(&self, board: &Vec<i8>, player: i8) -> i8;

    /// Input:
    ///     board: current board
    ///     player: current player (1 or -1)
    ///
    /// Returns:
    ///     canonicalBoard: returns canonical form of board. The canonical form
    ///                     should be independent of player. For e.g. in chess,
    ///                     the canonical form can be chosen to be from the pov
    ///                     of white. When the player is white, we can return
    ///                     board as is. When the player is black, we can invert
    ///                     the colors and return the board.
    fn get_canonical_form(&self, board: &Vec<i8>, player: i8) -> Vec<i8>;

    /// Input:
    ///     board: current board
    ///     pi: policy vector of size self.getActionSize()
    ///
    /// Returns:
    ///     symmForms: a list of [(board,pi)] where each tuple is a symmetrical
    ///                     form of the board and the corresponding pi vector. This
    ///                     is used when training the neural network from examples.
    fn get_symmetries(&self, board: &Vec<i8>, pi: &Vec<f32>) -> Vec<(Vec<i8>, Vec<f32>)>;

    /// Input:
    ///     board: current board
    ///
    /// Returns:
    ///     boardString: a quick conversion of board to a string format.
    ///                  Required by MCTS for hashing.
    fn string_representation(&self, board: &Vec<i8>) -> String;
}
