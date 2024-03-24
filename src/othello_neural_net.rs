use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
    },
    tensor::{
        activation::{log_softmax, relu, tanh},
        backend::{AutodiffBackend, Backend},
        Float, Tensor,
    },
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    bn3: BatchNorm<B, 2>,
    bn4: BatchNorm<B, 2>,
    fc1: Linear<B>,
    fc_bn1: BatchNorm<B, 1>,
    fc2: Linear<B>,
    fc_bn2: BatchNorm<B, 1>,
    fc3: Linear<B>,
    fc4: Linear<B>,
    dropout: Dropout,
    num_channels: usize,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "6")]
    board_x: i8,
    #[config(default = "6")]
    board_y: i8,
    #[config(default = "37")]
    action_size: usize,
    #[config(default = "6")]
    num_classes: usize,
    #[config(default = "512")]
    num_channels: usize,
    #[config(default = "512")]
    hidden_size: usize,
    #[config(default = "0.3")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, self.num_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([self.num_channels, self.num_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv3: Conv2dConfig::new([self.num_channels, self.num_channels], [3, 3]).init(device),
            conv4: Conv2dConfig::new([self.num_channels, self.num_channels], [3, 3]).init(device),
            bn1: BatchNormConfig::new(self.num_channels).init(device),
            bn2: BatchNormConfig::new(self.num_channels).init(device),
            bn3: BatchNormConfig::new(self.num_channels).init(device),
            bn4: BatchNormConfig::new(self.num_channels).init(device),
            fc1: LinearConfig::new(
                self.num_channels * ((self.board_x - 4) * (self.board_y - 4)) as usize,
                1024,
            )
            .init(device),
            fc_bn1: BatchNormConfig::new(1024).init(device),
            fc2: LinearConfig::new(1024, 512).init(device),
            fc_bn2: BatchNormConfig::new(512).init(device),
            fc3: LinearConfig::new(512, self.action_size).init(device),
            fc4: LinearConfig::new(512, 1).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            num_channels: self.num_channels,
        }
    }
}

impl<B: AutodiffBackend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3, Float>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch_size, board_x, board_y] = images.dims(); // batch_size x board_x x board_y

        let s = images.reshape([-1, 1, board_x as i32, board_y as i32]); // batch_size x 1 x board_x x board_y

        let s = relu(self.bn1.forward(self.conv1.forward(s)));
        let s = relu(self.bn2.forward(self.conv2.forward(s)));
        let s = relu(self.bn3.forward(self.conv3.forward(s)));
        let s = relu(self.bn4.forward(self.conv4.forward(s)));
        let s = s.reshape([
            -1,
            (self.num_channels * (board_x - 4) * (board_y - 4)) as i32,
        ]);

        let s = self.dropout.forward(relu(
            self.fc_bn1
                .forward(self.fc1.forward(s).reshape([batch_size, 1024, 1])),
        ));

        let s = self.dropout.forward(relu(
            self.fc_bn2.forward(
                self.fc2
                    .forward(s.reshape([batch_size as i32, -1]))
                    .reshape([batch_size, 512, 1]),
            ),
        ));

        let pi = self
            .fc3
            .forward(s.clone().reshape([batch_size as i32, -1]))
            .reshape([batch_size, board_x * board_y + 1]); // batch_size x action_size
        let v = self
            .fc4
            .forward(s.clone().reshape([batch_size as i32, -1]))
            .reshape([batch_size as i32, 1]); // batch_size x 1

        (log_softmax(pi, 1), tanh(v))
    }
}
