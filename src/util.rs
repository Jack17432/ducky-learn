extern crate ndarray;

use ndarray::Array2;
use std::error::Error;

/// Marker struct indicating a model that has not been fit.
pub struct Unfit;

/// Marker struct indicating a model that has been fit.
pub struct Fit;

/// Generates a one-hot encoding for a vector of integers.
///
/// # Arguments
///
/// * `input_array`: List of integers to be encoded. Each integer should be less than or equal to the maximum integer in the array.
///
/// Returns: `Array2<f64>` where each row represents the one-hot encoding of the corresponding integer from the input array. The columns represent the range of integers from the input array.
///
/// # Errors
///
/// Returns an error if the `input_array` is empty.
///
/// # Examples
///
/// ```
/// use ducky_learn::util::one_hot_encoding_vec;
/// use ndarray::prelude::*;
///
/// let input_array = vec![2, 0, 1];
/// let output_array = array![
///     [0., 0., 1.],
///     [1., 0., 0.],
///     [0., 1., 0.]
/// ];
///
/// assert_eq!(one_hot_encoding_vec(input_array).unwrap(), output_array);
/// ```
pub fn one_hot_encoding_vec<T: AsRef<[usize]>>(input_array: T) -> Result<Array2<f64>, Box<dyn Error>> {
    let input_array = input_array.as_ref();
    let max_val = match input_array.iter().max() {
        Some(&max) => max + 1,
        None => return Err("Empty input array".into()),
    };

    let mut encoding_array: Vec<Vec<f64>> = Vec::with_capacity(input_array.len());

    for &input_value in input_array {
        let mut row = vec![0.; max_val];
        row[input_value] = 1.;
        encoding_array.push(row);
    }

    let data: Vec<f64> = encoding_array.into_iter().flatten().collect();
    let n_row = input_array.len();
    let n_col = max_val;

    Array2::from_shape_vec((n_row, n_col), data).map_err(|err| err.into())
}
