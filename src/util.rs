extern crate ndarray;

use ndarray::prelude::*;

/// one hot encoding for a vector of integers.
///
/// # Arguments
///
/// * `input_array`: List of cats that are integers
/// * `max_val`: make value that the input array will contain or the size of the one hot encoding
///
/// returns: `Array2<f64>`
///
/// # Examples
///
/// ```
/// use ducky_learn::util::one_hot_encoding_vec;
/// use ndarray::prelude::*;
///
/// let input_array = vec![3, 1, 0];
/// let output_array = arr2(
///     &[[0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]]
/// );
///
/// assert_eq!(one_hot_encoding_vec(&input_array, 0), output_array);
/// ```
pub fn one_hot_encoding_vec(input_array: &Vec<usize>, max_val: u32) -> Array2<f64> {
    let mut encoding_array: Vec<Vec<f64>> = vec![vec![0.]; input_array.len()];

    for (idx, input_value) in input_array.iter().enumerate() {
        encoding_array[idx] = vec![0.; max_val as usize];
        encoding_array[idx].insert(*input_value, 1.);
    }

    let n_col = encoding_array.first().map_or(0, |row| row.len());
    let n_row = input_array.len();

    let mut data = Vec::new();
    for encoding_row in &encoding_array {
        data.extend_from_slice(encoding_row);
    }

    Array2::from_shape_vec((n_row, n_col), data).unwrap()
}
