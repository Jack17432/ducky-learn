extern crate ndarray;

use ndarray::prelude::*;

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
    ///
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