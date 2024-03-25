use std::collections::HashMap;
use std::io;

use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};

use crate::{
    coach::Coach, game::Game, n_net::NNetWrapper, neural_net::NeuralNet, othello::Othello,
};

mod arena;
mod board;
mod board_math;
mod coach;
mod game;
mod mcts;
mod n_net;
mod neural_net;
mod othello;
mod othello_neural_net;

fn main() {
    let mut args: HashMap<String, String> = HashMap::new();
    args.insert("numIters".to_string(), "1000".to_owned());
    args.insert("numEps".to_string(), "100".to_owned());
    args.insert("tempThreshold".to_string(), "15".to_owned());
    args.insert("updateThreshold".to_string(), "0.6".to_owned());
    args.insert("maxlenOfQueue".to_string(), "200000".to_owned());
    args.insert("numMCTSSims".to_string(), "25".to_owned());
    args.insert("arenaCompare".to_string(), "40".to_owned());
    args.insert("cpuct".to_string(), "1".to_owned());
    args.insert("checkpoint".to_string(), "./temp/".to_owned());
    args.insert("load_model".to_string(), "true".to_owned());
    args.insert("load_folder".to_string(), "temp".to_owned());
    args.insert("load_folder_file".to_string(), "best.pth.tar".to_owned());
    args.insert("load_examples_file".to_string(), "checkpoint_3.pth.tar".to_owned());
    args.insert(
        "numItersForTrainExamplesHistory".to_string(),
        "20".to_owned(),
    );
    args.insert("verbose".to_owned(), "false".to_owned());

    println!("Loading {:?}...", "Othello");
    let g = Othello::new(6);

    println!("Loading {:?}...", "LibTorch");
    let device = LibTorchDevice::Cuda(0);
    type MyBackend = Autodiff<LibTorch>;
    let mut nnet: NNetWrapper<MyBackend, Othello> = NNetWrapper::new(g.clone(), device);

    let load_model = args.get("load_model").unwrap().parse::<bool>().unwrap();
    if load_model {
        let folder = args.get("load_folder").unwrap();
        let filename = args.get("load_folder_file").unwrap();
        println!("Loading checkpoint \"{folder}/{filename}\"...");
        nnet.load_checkpoint(folder, filename);
    } else {
        println!("Not loading a checkpoint!");
    }

    println!("Loading the Coach...");
    let mut c = Coach::new(g.clone(), nnet, device, args);

    // TODO potential loading in of training examples
    if load_model {
        println!("Loading 'trainExamples' from file...");
        c.load_train_examples();
    }

    println!("Starting the learning process 🎉");
    c.learn();
    println!("Going to wait...");
    io::stdin().read_line(&mut String::new()).unwrap();
}
