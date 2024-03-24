use burn::tensor::backend::AutodiffBackend;

use crate::game::Game;

pub trait NeuralNet<B: AutodiffBackend, G: Game> {
    /// This class specifies the base NeuralNet class. To define your own neural
    /// network, subclass this class and implement the functions below. The neural
    /// network does not consider the current player, and instead only deals with
    /// the canonical form of the board.
    ///
    /// See othello/NNet.py for an example implementation.

    fn new(game: G, device: B::Device) -> Self;

    /// This function trains the neural network with examples obtained from
    /// self-play.
    ///
    /// Input:
    ///     examples: a list of training examples, where each example is of form
    ///               (board, pi, v). pi is the MCTS informed policy vector for
    ///               the given board, and v is its value. The examples has
    ///               board in its canonical form.
    fn train(&self, examples: &Vec<(Vec<i8>, Vec<f32>, i8)>);

    /// Input:
    /// board: current board in its canonical form.
    ///
    /// Returns:
    ///     pi: a policy vector for the current board- a numpy array of length
    ///         game.getActionSize
    ///     v: a float in [-1,1] that gives the value of the current board
    fn predict(&self, board: &Vec<i8>) -> (Vec<f32>, f32);

    /// Saves the current neural network (with its parameters) in
    /// folder/filename
    fn save_checkpoint(&self, folder: &str, filename: &str);

    /// Loads parameters of the neural network from folder/filename
    fn load_checkpoint(&mut self, folder: &str, filename: &str);
}
