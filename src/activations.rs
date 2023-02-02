extern crate ndarray;

use ndarray::prelude::*;

/// Relu activation function for 1d array
///
/// More info: https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
///
/// # Arguments
///
/// * `input_array`: 1d array
///
/// returns: `Array1<f64>`
///
/// # Examples
///
/// ```
/// use ducky_learn::activations::*;
/// use ndarray::arr1;
///
/// let input_array = arr1(&[-1.3456435325242, -32145324321., -132432888.]);
/// assert_eq!(relu_1d(input_array), arr1(&[0., 0., 0.]));
/// ```
pub fn relu_1d(input_array: Array1<f64>) -> Array1<f64> {
    input_array.map(|value| value.max(0.))
}

/// Softmax activation function for 1d array. Note that you can run into NaN issue if values are
/// < -1000 or > 1000 (https://users.rust-lang.org/t/watch-out-for-nans/70016)
///
/// More info: https://deepai.org/machine-learning-glossary-and-terms/softmax-layer#:~:text=The%20softmax%20function%20is%20a,can%20be%20interpreted%20as%20probabilities.
///
/// # Arguments
///
/// * `input_array`: 1d array
///
/// returns: `Array1<f64>`
///
/// # Examples
///
/// ```
/// use ducky_learn::activations::*;
/// use ndarray::arr1;
///
/// let input_array = arr1(&[0., 1., -1., 0.01, -0.1]);
/// assert_eq!(softmax_1d(input_array),
///            arr1(&[0.16663753690463112, 0.4529677885070323, 0.0613025239546613, 0.16831227199301688, 0.15077987864065834]));
/// ```
pub fn softmax_1d(input_array: Array1<f64>) -> Array1<f64> {
    let sum_exp_input_array = input_array
        .map(|value| value.exp())
        .sum();

    input_array.map(|value| value.exp() / sum_exp_input_array)
}