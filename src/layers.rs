extern crate ndarray;
extern crate ndarray_rand;

use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::sync::RwLock;

pub trait Layer1d {
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
    ///                 |x| x.map(|i| 1f64), // Derivative of Activation function
    ///                 arr2(&[[1., 1.], [1., 1.]]), // 2x2 array
    ///                 arr1(&[1., 1.]) // len 2
    ///             );
    ///
    /// let output = layer.pass(arr1(&[1., 1.]));
    ///
    /// ```
    fn pass(&self, input_array: Array1<f64>) -> (Array1<f64>, Array1<f64>); // TODO: update doc
}

pub struct Dense1d {
    activation: fn(Array1<f64>) -> Array1<f64>,
    deriv_activation: fn(Array1<f64>) -> Array1<f64>,
    weights: RwLock<Array2<f64>>,
    bias: RwLock<Array1<f64>>,
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
    ///                 |x| x.map(|i| 1f64), // Derivative of Activation function
    ///                 arr2(&[[1., 1.], [1., 1.]]), // 2x2 array
    ///                 arr1(&[1., 1.]) // len 2
    ///             );
    /// ```
    pub fn from(
        activation: fn(Array1<f64>) -> Array1<f64>,
        deriv_activation: fn(Array1<f64>) -> Array1<f64>,
        weights: Array2<f64>,
        bias: Array1<f64>,
    ) -> Self {
        Self {
            activation,
            deriv_activation,
            weights: RwLock::new(weights),
            bias: RwLock::new(bias),
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
    /// let layer = Dense1d::new(5, 10, |x| x, |x| x);
    /// let input_array = arr1(&[
    ///     1., 1., 1., 1., 1.
    /// ]);
    ///
    /// layer.pass(input_array);
    /// ```
    pub fn new(
        input_size: usize,
        layer_size: usize,
        activation_fn: fn(Array1<f64>) -> Array1<f64>,
        deriv_activation_fn: fn(Array1<f64>) -> Array1<f64>,
    ) -> Self {
        Self {
            activation: activation_fn,
            deriv_activation: deriv_activation_fn,
            weights: RwLock::new(Array2::random(
                (layer_size, input_size),
                Uniform::new(-1., 1.),
            )),
            bias: RwLock::new(Array1::random(layer_size, Uniform::new(-1., 1.))),
        }
    }
}

impl Layer1d for Dense1d {
    fn pass(&self, input_array: Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let weights = self.weights.read().unwrap();
        let bias = self.bias.read().unwrap();

        assert_eq!(
            weights.shape()[1],
            input_array.shape()[0],
            "Layer input size is {}, \
            Layer was given size of {}",
            weights.shape()[1],
            input_array.shape()[0]
        );

        let z = weights.dot(&input_array) + &*bias;
        let a = (self.activation)(z.clone());
        (z, a)
    }
}


#[cfg(test)]
mod layers_tests {
    use super::*;
    use ndarray::*;
    use crate::activations::*;

    #[test]
    fn dense1d_pass_arr1_1() {
        let layer = Dense1d::from(
            |x| x,
            |x| x,
            arr2(&[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
            arr1(&[1., 1., 1.]),
        );
        let input_array = arr1(&[1., 1., 1.]);

        assert_eq!(layer.pass(input_array).1, arr1(&[4., 4., 4.]))
    }

    #[test]
    fn dense1d_pass_arr1_2() {
        let layer = Dense1d::from(
            |x| x,
            |x| x,
            arr2(&[
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            ]),
            arr1(&[1., 1., 1.]),
        );
        let input_array = arr1(&[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]);

        assert_eq!(layer.pass(input_array).1, arr1(&[13.0, 13.0, 13.0]))
    }

    #[test]
    #[should_panic]
    fn dense1d_pass_arr1_diff_size() {
        let layer = Dense1d::from(
            |x| x,
            |x| x,
            arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]),
            arr1(&[0., 0.]),
        );
        let input_array = arr1(&[1.]);

        layer.pass(input_array);
    }

    #[test]
    fn dense1d_new() {
        let layer = Dense1d::new(5, 10, |x| x, |x| x);

        let input_array = arr1(&[1., 1., 1., 1., 1.]);

        layer.pass(input_array);
    }

    #[test]
    fn dense1d_activation() {
        let layer = Dense1d::from(
            relu_1d,
            deriv_relu_1d,
            arr2(&[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
            arr1(&[-10., -10., 1.]),
        );
        let input_array = arr1(&[1., 1., 1.]);

        assert_eq!(layer.pass(input_array).1, arr1(&[0., 0., 4.]))
    }
}
