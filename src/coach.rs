use burn::tensor::backend::AutodiffBackend;
use rand::distributions::Distribution;
use rand::seq::SliceRandom;
use rand::{distributions::WeightedIndex, thread_rng};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::time::SystemTime;

use crate::arena::Arena;
use crate::othello::Othello;
use crate::{game::Game, mcts::MCTS, n_net::NNetWrapper, neural_net::NeuralNet};

pub struct Coach<G, B>
where
    G: Game,
    B: AutodiffBackend,
{
    game: G,
    nnet: NNetWrapper<B, G>,
    pnet: NNetWrapper<B, G>,
    args: HashMap<String, String>,
    mcts: MCTS<G, B>,
    training_examples_history: VecDeque<Vec<(Vec<i8>, Vec<f32>, i8)>>,
    skip_first_self_play: bool,
}

impl<G: Clone, B> Coach<G, B>
where
    G: Game,
    B: AutodiffBackend,
{
    pub fn new(
        game: G,
        nnet: NNetWrapper<B, G>,
        device: B::Device,
        args: HashMap<String, String>,
    ) -> Self {
        Coach {
            game: game.clone(),
            nnet: nnet.clone(),
            pnet: NNetWrapper::new(game.clone(), device),
            args: args.clone(),
            mcts: MCTS::new(game.clone(), nnet.clone(), args.clone()),
            training_examples_history: VecDeque::new(),
            skip_first_self_play: false,
        }
    }

    fn execute_episode(&mut self) -> Vec<(Vec<i8>, Vec<f32>, i8)> {
        let mut train_examples = Vec::<(Vec<i8>, Vec<f32>, i8)>::new();
        let mut board = self.game.get_init_board().clone();
        let mut cur_player = 1;
        let mut episode_step = 0;

        loop {
            episode_step += 1;
            let canonical_board = self.game.get_canonical_form(&board, cur_player);
            let temp_threshold = self
                .args
                .get("tempThreshold")
                .unwrap()
                .parse::<i32>()
                .unwrap();
            let temp = if episode_step < temp_threshold { 1 } else { 0 };

            let pi = self.mcts.get_action_prob(&canonical_board, temp);
            let sym = self.game.get_symmetries(&canonical_board, &pi);
            for s in sym {
                let b = s.0;
                let p = s.1;
                let tup = (b, p, cur_player);
                train_examples.push(tup);
            }

            let weighted_index = WeightedIndex::new(&pi).unwrap();
            let mut rng = thread_rng();
            let action = weighted_index.sample(&mut rng);
            let next_state = self.game.get_next_state(&board, cur_player, action as u8);
            board = next_state.0;
            cur_player = next_state.1;

            let r = self.game.get_game_ended(&board, cur_player);

            if r != 0 {
                for tup in train_examples.iter_mut() {
                    tup.2 = r * ((-1 as i8).pow(if tup.2 != cur_player { 1 } else { 0 }));
                }
                return train_examples;
            }
        }
    }

    pub fn learn(&mut self) {
        let num_iters = self.args.get("numIters").unwrap().parse::<i32>().unwrap();
        let maxlen_of_queue = self
            .args
            .get("maxlenOfQueue")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let num_eps = self.args.get("numEps").unwrap().parse::<i32>().unwrap();
        let num_iters_for_train_examples_history = &self
            .args
            .get("numItersForTrainExamplesHistory")
            .unwrap()
            .parse::<usize>()
            .unwrap();

        for i in 1..num_iters + 1 {
            // bookkeeping
            println!("Starting iter #{:?}", i);
            let now = SystemTime::now();

            // examples of the iteration
            if !&self.skip_first_self_play || i > 1 {
                println!("Not skipping first self play");
                let mut iteration_train_examples: VecDeque<Vec<(Vec<i8>, Vec<f32>, i8)>> =
                    VecDeque::with_capacity(maxlen_of_queue);

                for _j in 0..num_eps {
                    self.mcts = MCTS::new(self.game.clone(), self.nnet.clone(), self.args.clone());
                    iteration_train_examples.push_back(self.execute_episode());
                }

                // save the iteration examples to the history
                self.training_examples_history
                    .append(&mut iteration_train_examples);
            }

            if &self.training_examples_history.len() > num_iters_for_train_examples_history {
                println!("WARNING: Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {:?}", &self.training_examples_history.len());
                let _ = &self.training_examples_history.pop_front();
            }

            // backup history to a file
            // Nota Bene NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_examples(i - 1);

            // shuffle examples before training
            let mut train_examples = Vec::new();
            for e in self.training_examples_history.clone() {
                train_examples.extend(e);
            }
            train_examples.shuffle(&mut thread_rng());

            // train new network, keeping a copy of the old one
            self.nnet
                .save_checkpoint(self.args.get("checkpoint").unwrap(), "temp.pth.tar");
            self.pnet
                .load_checkpoint(self.args.get("checkpoint").unwrap(), "temp.pth.tar");
            let mut pmcts = MCTS::new(self.game.clone(), self.pnet.clone(), self.args.clone());

            self.nnet.train(&train_examples);
            let mut nmcts = MCTS::new(self.game.clone(), self.nnet.clone(), self.args.clone());

            println!("PITTING AGAINST PREVIOUS VERSION");
            let lambda1 = |x: &Vec<i8>| {
                let pi = pmcts.get_action_prob(x, 0);
                let max = pi.iter().cloned().fold(0. / 0., f32::max);
                let mut best_as = Vec::new();
                for i in 0..pi.len() {
                    if pi[i] == max {
                        best_as.push(i);
                    }
                }
                return best_as.first().unwrap().to_owned();
            };
            let lambda2 = |x: &Vec<i8>| {
                let pi = nmcts.get_action_prob(x, 0);
                let max = pi.iter().cloned().fold(0. / 0., f32::max);
                let mut best_as = Vec::new();
                for i in 0..pi.len() {
                    if pi[i] == max {
                        best_as.push(i);
                    }
                }
                return best_as.first().unwrap().to_owned();
            };
            let mut arena = Arena::new(lambda1, lambda2, &self.game, Othello::display);
            let arena_compare = self
                .args
                .get("arenaCompare")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let verbose = self.args.get("verbose").unwrap().parse::<bool>().unwrap();
            let results = arena.play_games(arena_compare, verbose);
            let pwins = results.0;
            let nwins = results.1;
            let draws = results.2;

            println!(
                "NEW/PREV WINS : {:?} / {:?} ; DRAWS : {:?}",
                nwins, pwins, draws
            );
            let update_threshold = self
                .args
                .get("updateThreshold")
                .unwrap()
                .parse::<f32>()
                .unwrap();
            let checkpoint = self.args.get("checkpoint").unwrap();
            if pwins + nwins == 0 || (nwins as f32 / (pwins + nwins) as f32) < update_threshold {
                println!("REJECTING NEW MODEL");
                self.nnet.load_checkpoint(&checkpoint, "temp.pth.tar");
            } else {
                println!("ACCEPTING NEW MODEL");
                self.nnet
                    .save_checkpoint(&checkpoint, &self.get_checkpoint_file(i.to_string()));
                self.nnet.save_checkpoint(&checkpoint, "best.pth.tar");
            }

            match now.elapsed() {
                Ok(elapsed) => {
                    println!("This iteration took {} seconds", elapsed.as_secs_f32());
                }
                Err(e) => {
                    // an error occurred!
                    println!("Error: {e:?}");
                }
            }
        }
    }

    fn get_checkpoint_file(&self, iteration: String) -> String {
        return format!("checkpoint_{iteration}.pth.tar");
    }

    fn save_train_examples(&self, iteration: i32) {
        let folder = self.args.get("checkpoint").unwrap();
        let filename = self.get_checkpoint_file(iteration.to_string());
        let file_path = format!("{folder}/{filename}");
        fs::create_dir_all(folder).expect("Should be able to create path");
        let mut buffer = File::create(file_path).unwrap();

        let serialized =
            serde_pickle::to_vec(&self.training_examples_history, Default::default()).unwrap();
        let _ = buffer.write_all(&serialized);
    }

    pub fn load_train_examples(&mut self) {
        let folder = self.args.get("checkpoint").unwrap();
        let filename = self.args.get("load_examples_file").unwrap();
        let file_path = format!("{folder}/{filename}");
        let serialized = fs::read(file_path).unwrap();

        self.training_examples_history = serde_pickle::from_slice::<VecDeque<Vec<(Vec<i8>, Vec<f32>, i8)>>>(&serialized, Default::default()).unwrap();
    }
}
