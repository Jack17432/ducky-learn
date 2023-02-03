extern crate ndarray;
extern crate ndarray_rand;

use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub trait FeedForward1d {
    /// Feeds forward the 1d array through the layer.
    ///
    /// # Arguments
    ///
    /// * `input_array`: Has to be the same size as the input size of the layer else will panic
    ///
    /// returns: `Array1<f64>`
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::layers::*;
    /// use ndarray::{arr1, arr2};
    ///
    /// let layer = Dense1d::from(
    ///                 |x| x, // Activation function that is does nothing
    ///                 arr2(&[[1., 1.], [1., 1.]]), // 2x2 array
    ///                 arr1(&[1., 1.]) // len 2
    ///             );
    ///
    /// let output = layer.pass(arr1(&[1., 1.]));
    ///
    /// ```
    fn pass(&self, input_array: Array1<f64>) -> Array1<f64>;
}

pub struct Dense1d {
    activation: fn(Array1<f64>) -> Array1<f64>,
    weights: Array2<f64>,
    bias: Array1<f64>
}

impl Dense1d {
    /// Create Dense1d layer with full control over every part of the layer
    ///
    /// # Arguments
    ///
    /// * `activation`: Activation function of whole 1d array
    /// * `weights`: 2d array that has to be of shape( output, input )
    /// * `bias`: 1d array of basis that has to be the size of the output
    ///
    /// returns: `Dense1d`
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::layers::*;
    /// use ndarray::{arr1, arr2};
    ///
    /// let layer = Dense1d::from(
    ///                 |x| x, // Activation function that is does nothing
    ///                 arr2(&[[1., 1.], [1., 1.]]), // 2x2 array
    ///                 arr1(&[1., 1.]) // len 2
    ///             );
    /// ```
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

    /// Create randomly set weights and bias's for the dense1d layer.
    /// Creates weights and bias's using a normal distribution from -1. -> 1.
    ///
    /// # Arguments
    ///
    /// * `input_size`: size of input array
    /// * `layer_size`: number of nodes in the layer
    /// * `activation_fn`: activation function for the layer
    ///
    /// returns: `Dense1d`
    ///
    /// # Examples
    ///
    /// ```
    /// use ducky_learn::layers::*;
    /// use ndarray::{arr1, arr2};
    ///
    /// let layer = Dense1d::new(5, 10, |x| x);
    /// let input_array = arr1(&[
    ///     1., 1., 1., 1., 1.
    /// ]);
    ///
    /// layer.pass(input_array);
    /// ```
    pub fn new(
        input_size: usize,
        layer_size: usize,
        activation_fn: fn(Array1<f64>) -> Array1<f64>
    ) -> Self {
        Self {
            activation: activation_fn,
            weights: Array2::random((layer_size, input_size),
                                    Uniform::new(-1., 1.)),
            bias: Array1::random(layer_size,
                                 Uniform::new(-1., 1.))
        }
    }
}

impl FeedForward1d for Dense1d {
    fn pass(&self, input_array: Array1<f64>) -> Array1<f64> {
        assert_eq!(self.weights.shape()[1], input_array.shape()[0],
            "Layer input size is {}, \
            Layer was given size of {}",
            self.weights.shape()[1], input_array.shape()[0]
        );

        (self.activation)(self.weights.dot(&input_array) + &self.bias)
    }
}