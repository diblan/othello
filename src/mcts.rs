use std::{collections::HashMap, marker::PhantomData};

use burn::tensor::backend::AutodiffBackend;
use rand::seq::SliceRandom;

use crate::{game::Game, n_net::NNetWrapper, neural_net::NeuralNet};

pub struct MCTS<G: Game, B: AutodiffBackend> {
    game: G,
    nnet: NNetWrapper<B, G>,
    args: HashMap<String, String>,
    phantom: PhantomData<B>,
    qsa: HashMap<(String, usize), f32>,
    nsa: HashMap<(String, usize), f32>,
    ns: HashMap<String, usize>,
    ps: HashMap<String, Vec<f32>>,
    es: HashMap<String, i8>,
    vs: HashMap<String, Vec<u8>>,
}

impl<G: Game, B: AutodiffBackend> MCTS<G, B> {
    pub fn new(game: G, nnet: NNetWrapper<B, G>, args: HashMap<String, String>) -> Self {
        MCTS {
            game,
            nnet,
            args,
            phantom: PhantomData,
            qsa: HashMap::new(),
            nsa: HashMap::new(),
            ns: HashMap::new(),
            ps: HashMap::new(),
            es: HashMap::new(),
            vs: HashMap::new(),
        }
    }

    /// This function performs numMCTSSims simulations of MCTS starting from
    /// canonicalBoard.
    ///
    /// Returns:
    ///     probs: a policy vector where the probability of the ith action is
    ///            proportional to Nsa[(s,a)]**(1./temp)
    pub fn get_action_prob(&mut self, canonical_board: &Vec<Vec<i8>>, temp: u8) -> Vec<f32> {
        for _i in 0..self.args.get("numMCTSSims").unwrap().parse().unwrap() {
            self.search(canonical_board);
        }

        let s = self.game.string_representation(canonical_board);
        let mut counts = Vec::<f32>::with_capacity(self.game.get_action_size());
        for i in 0..self.game.get_action_size() {
            let nsa_key = (s.clone(), i);
            let nsa_value = self.nsa.get(&nsa_key).unwrap_or(&0.);
            counts.push(*nsa_value);
        }

        if temp == 0 {
            let max = counts.iter().cloned().fold(0. / 0., f32::max);
            let mut best_as = Vec::new();
            for i in 0..counts.len() {
                if counts[i] == max {
                    best_as.push(i);
                }
            }
            let best_a = *best_as.choose(&mut rand::thread_rng()).unwrap();
            let mut probs = vec![0.; counts.len()];
            probs[best_a] = 1.;
            return probs;
        }

        for i in counts.iter_mut() {
            *i = i.powf(1. / temp as f32);
        }
        let counts_sum = counts.iter().cloned().fold(0., |acc, x| acc + x);
        for i in counts.iter_mut() {
            *i /= counts_sum;
        }
        return counts;
    }

    /// This function performs one iteration of MCTS. It is recursively called
    /// till a leaf node is found. The action chosen at each node is one that
    /// has the maximum upper confidence bound as in the paper.
    ///
    /// Once a leaf node is found, the neural network is called to return an
    /// initial policy P and a value v for the state. This value is propagated
    /// up the search path. In case the leaf node is a terminal state, the
    /// outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
    /// updated.
    ///
    /// NOTE: the return values are the negative of the value of the current
    /// state. This is done since v is in [-1,1] and if v is the value of a
    /// state for the current player, then its value is -v for the other player.
    ///
    /// Returns:
    ///     v: the negative of the value of the current canonicalBoard
    fn search(&mut self, canonical_board: &Vec<Vec<i8>>) -> f32 {
        let s = self.game.string_representation(canonical_board);

        if !self.es.contains_key(&s) {
            self.es
                .insert(s.clone(), self.game.get_game_ended(canonical_board, 1));
        }
        if *self.es.get(&s).unwrap() != 0 as i8 {
            // terminal node
            return -self.es.get(&s).unwrap() as f32;
        }

        if !self.ps.contains_key(&s) {
            // leaf node
            let output = self.nnet.predict(canonical_board);
            let pi = output.0;
            let v = output.1;
            let valids = self.game.get_valid_moves(canonical_board, 1);
            // a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
            let mut pi = pi
                .iter()
                .zip(valids.iter())
                .map(|(x, y)| x * *y as f32)
                .collect::<Vec<f32>>();
            let sum_ps_s = pi.iter().sum::<f32>();
            if sum_ps_s > 0.0 {
                for i in pi.iter_mut() {
                    *i /= sum_ps_s;
                }
            } else {
                // if all valid moves were masked make all valid moves equally probable

                // NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                // If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                println!("All valid moves were masked, doing a workaround.");
                pi = pi
                    .iter()
                    .zip(valids.iter())
                    .map(|(x, y)| x + *y as f32)
                    .collect::<Vec<f32>>();
                let sum = pi.iter().sum::<f32>();
                for i in pi.iter_mut() {
                    *i /= sum;
                }
            }

            self.ps.insert(s.clone(), pi);
            self.vs.insert(s.clone(), valids);
            self.ns.insert(s.clone(), 0);
            return -v;
        }

        let valids = self.vs.get(&s).unwrap();
        let mut cur_best = std::f32::NEG_INFINITY;
        let mut best_act = -1;

        // pick the action with the highest upper confidence bound
        for a in 0..self.game.get_action_size() {
            if *valids.get(a).unwrap() > 0 {
                let qsa_key = (s.clone(), a.clone());
                let cpuct = self.args.get("cpuct").unwrap().parse::<f32>().unwrap();
                let ps_value = self.ps.get(&s).unwrap().get(a).unwrap();
                let sqrt_ns = f32::sqrt(*self.ns.get(&s).unwrap() as f32);
                let eps = 1e-8;
                let u;
                if self.qsa.contains_key(&qsa_key) {
                    let qsa_value = self.qsa.get(&qsa_key).unwrap();
                    let nsa_value = self.nsa.get(&qsa_key).unwrap();
                    u = qsa_value + cpuct * ps_value * sqrt_ns / (1.0 + nsa_value);
                } else {
                    u = cpuct * ps_value * f32::sqrt(*self.ns.get(&s).unwrap() as f32 + eps as f32);
                }
                if u > cur_best {
                    cur_best = u;
                    best_act = a as isize;
                }
            }
        }

        let a = best_act;
        let next_state = self.game.get_next_state(canonical_board, 1, a as u8);
        let next_s = self.game.get_canonical_form(&next_state.0, next_state.1);

        let v = self.search(&next_s);

        let qsa_key = (s.clone(), a as usize);
        if self.qsa.contains_key(&qsa_key) {
            let nsa_value = self.nsa.get(&qsa_key).unwrap();
            let qsa_value = self.qsa.get(&qsa_key).unwrap();
            let new_qsa_value = (nsa_value * qsa_value + v) / (nsa_value + 1.0);
            self.qsa.insert(qsa_key.clone(), new_qsa_value);
            if let Some(x) = self.nsa.get_mut(&qsa_key) {
                *x += 1.0;
            }
        } else {
            self.qsa.insert(qsa_key.clone(), v);
            self.nsa.insert(qsa_key.clone(), 1.0);
        }

        if let Some(x) = self.ns.get_mut(&s) {
            *x += 1;
        }
        return -v;
    }
}
