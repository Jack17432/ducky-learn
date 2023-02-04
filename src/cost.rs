use ndarray::Array1;

// TODO: Documentation
pub fn mean_squared_error(observed_array: Array1<f64>, predicted_array: Array1<f64>) -> Array1<f64> {
    (&observed_array - &predicted_array).mapv(|value| value.powi(2))
}

// TODO: Documentation
pub fn deriv_mean_squared_error(observed_array: Array1<f64>, predicted_array: Array1<f64>) -> Array1<f64> {
    -2f64 * (&observed_array - &predicted_array)
}