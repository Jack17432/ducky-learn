use ndarray::prelude::*;

pub trait FeedForward1d {
    fn pass(&self, input_array: Array1<f64>) -> Array1<f64>;
}

pub struct Dense1d {
    activation: fn(Array1<f64>) -> Array1<f64>,
    weights: Array2<f64>,
    bias: Array1<f64>
}

impl Dense1d {
    pub fn from(
        activation: fn(Array1<f64>) -> Array1<f64>,
        weights: Array2<f64>,
        bias: Array1<f64>,
    ) -> Self {
        Self {
            activation,
            weights,
            bias
        }
    }
}

impl FeedForward1d for Dense1d {
    fn pass(&self, input_array: Array1<f64>) -> Array1<f64> {
        assert_eq!(self.weights.shape()[1], input_array.shape()[0],
            "Layer input size is {}, \
            given shape is size of {}",
            self.weights.shape()[1], input_array.shape()[0]
        );

        self.weights.dot(&input_array) + &self.bias
    }
}