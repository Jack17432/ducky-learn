extern crate ndarray;

use ndarray::prelude::*;

pub fn one_hot_encoding_vec(input_array: &Vec<f64>, max_val: u32) -> Array2<f64> {
    let mut encoding_array: Vec<Vec<f64>> = vec![vec![0.]; input_array.len()];

    for (idx, input_value) in input_array.iter().enumerate() {
        encoding_array[idx] = vec![0.; max_val as usize];
        encoding_array[idx].insert(*input_value as usize, 1.);
    }

    let n_col = encoding_array.first().map_or(0, |row| row.len());
    let n_row = input_array.len();

    let mut data = Vec::new();
    for encoding_row in &encoding_array {
        data.extend_from_slice(encoding_row);
    }

    Array2::from_shape_vec((n_row, n_col), data).unwrap()
}