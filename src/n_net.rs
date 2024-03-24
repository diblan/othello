use std::{fs, marker::PhantomData};

use crate::{
    game::Game,
    neural_net::NeuralNet,
    othello_neural_net::{Model, ModelConfig},
};
use burn::{
    module::Module,
    optim::{AdamConfig, GradientsParams},
    tensor::{backend::AutodiffBackend, Data, Float, Shape, Tensor},
};
use burn::{
    optim::Optimizer,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use rand::{distributions::Uniform, Rng};

#[derive(Clone)]
pub struct NNetWrapper<B: AutodiffBackend, G: Game> {
    lr: f64,
    // dropout: f64,
    epochs: i32,
    batch_size: usize,
    // cuda: bool,
    // num_channels: i32,
    device: B::Device,
    nnet: Model<B>,
    phantom: PhantomData<G>,
}

impl<G: Game, B: AutodiffBackend> NeuralNet<B, G> for NNetWrapper<B, G> {
    fn new(game: G, device: B::Device) -> NNetWrapper<B, G> {
        let _board_x = game.get_board_size().0;
        let _board_y = game.get_board_size().1;
        let _action_size = game.get_action_size();
        let mc = ModelConfig::new();
        let nnet = mc.init::<B>(&device);
        NNetWrapper {
            lr: 0.001,
            // dropout: 0.3,
            epochs: 10,
            batch_size: 64,
            // cuda: true,
            // num_channels: 512,
            device,
            nnet,
            phantom: PhantomData,
        }
    }

    fn train(&self, examples: &Vec<(Vec<i8>, Vec<f32>, i8)>) {
        let mut optimizer = AdamConfig::new().init();

        for epoch in 0..self.epochs {
            println!("EPOCH ::: {:?}", epoch + 1);

            let batch_count = examples.len() / self.batch_size;

            for _i in 0..batch_count {
                let rand_generator = rand::thread_rng();
                let distr = Uniform::new(0, examples.len());
                let sample_ids: Vec<usize> = rand_generator
                    .sample_iter(distr)
                    .take(self.batch_size)
                    .collect();

                // TODO hard coded dimension 6 is not OK
                let mut boards_array: [[[f64; 6]; 6]; 64] = [[[1.0; 6]; 6]; 64];
                let mut pis_vec = Vec::<f32>::with_capacity(64 * 6 * 6);
                let mut vs_vec: Vec<f32> = Vec::<f32>::with_capacity(64);
                for i in 0..sample_ids.len() {
                    let example = examples.get(sample_ids[i]).unwrap();
                    let board = example.clone().0;
                    for j in 0..6 {
                        for k in 0..6 {
                            boards_array[i][j][k] =
                                f64::from(board.get(j * 6 + k).unwrap().to_owned());
                        }
                    }

                    let mut pi: Vec<f32> = example.1.clone();
                    pis_vec.append(&mut pi);

                    let v: i8 = example.2;
                    vs_vec.push(v as f32);
                    // vs_vec.append(&mut v); can be used later to make 2 dimensional [64, 1]
                }

                let boards_data: Data<_, 3> = Data::<f64, 3>::from(boards_array).convert();
                let mut boards_tensor: Tensor<B, 3> =
                    Tensor::<B, 3>::from_data(boards_data, &self.device);
                boards_tensor = boards_tensor.require_grad();

                let pis_shape = Shape::new([64, 37]);
                let pis_data: Data<_, 2> = Data::<f32, 2>::new(pis_vec, pis_shape).convert();
                let target_pis: Tensor<B, 2> = Tensor::<B, 2>::from_data(pis_data, &self.device);

                let vs_shape = Shape::new([64]);
                let vs_data: Data<_, 1> = Data::<f32, 1>::new(vs_vec, vs_shape).convert();
                let target_vs: Tensor<B, 1> = Tensor::<B, 1>::from_data(vs_data, &self.device);

                // compute output
                let output = &self.nnet.forward(boards_tensor);
                let l_pi = &self.loss_pi(&target_pis, output.0.clone());
                let l_v = &self.loss_v(&target_vs, output.1.clone());
                let total_loss = l_pi.clone() - l_v.clone();

                // let total_loss = total_loss.require_grad();
                let grads = total_loss.backward();
                let grads = GradientsParams::from_grads(grads, &self.nnet);
                optimizer.step(self.lr, self.nnet.clone(), grads);
            }
        }
    }

    /// board: np array with board
    fn predict(&self, board: &Vec<i8>) -> (Vec<f32>, f32) {
        // timing
        // start = time.time()

        let mut floats: Vec<f32> = Vec::with_capacity(36);
        for x in board {
            floats.push(x.clone() as f32);
        }
        let shape = Shape::new([1, 6, 6]);
        let data = Data::<f32, 3>::new(floats, shape).convert();
        let b: Tensor<B, 3, Float> = Tensor::<B, 3>::from_data(data, &self.device);
        let model = self.nnet.clone();
        let model = model.no_grad();
        let output: (Tensor<B, 2>, Tensor<B, 2>) = model.forward(b);

        let pi = output.0.exp().to_data().convert::<f32>().value;
        let v = output.1.to_data().convert::<f32>().value[0];

        // print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return (pi, v);
    }

    fn save_checkpoint(&self, folder: &str, filename: &str) {
        let file_path = format!("{folder}/{filename}");
        fs::create_dir_all(folder).expect("Should be able to create path");

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.nnet
            .clone()
            .save_file(file_path, &recorder)
            .expect("Should be able to save the model");
    }

    fn load_checkpoint(&mut self, folder: &str, filename: &str) {
        let file_path = format!("{folder}/{filename}");

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.nnet = self
            .nnet
            .clone()
            .load_file(file_path, &recorder, &self.device)
            .expect("Should be able to load the model weights from the provided file");
    }
}

impl<B: AutodiffBackend, G: Game> NNetWrapper<B, G> {
    fn loss_pi(&self, targets: &Tensor<B, 2>, outputs: Tensor<B, 2>) -> Tensor<B, 1> {
        let product = targets.clone().mul(outputs);
        let sum = -product.sum();
        let div = targets.dims()[0] as u32;
        let result = sum.div_scalar(div);
        return result;
    }

    fn loss_v(&self, targets: &Tensor<B, 1>, outputs: Tensor<B, 2>) -> Tensor<B, 1> {
        let outputs_1d = outputs.reshape([-1]);
        let min = targets.clone() - outputs_1d;
        let quad = min.powi_scalar(2);
        let sum = quad.sum();
        let div = targets.dims()[0] as u32;
        let result = sum.div_scalar(div);
        return result;
    }
}
