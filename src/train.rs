use std::iter::zip;
use ndarray::Array1;
use super::layers::*;

// TODO: Documentation
pub fn train_pass<L>(model: Vec<L>, input_array: Array1<f64>, correct_answer_array: Array1<f64>)
where
    L: Layer1d
{
    // Start off with a forward pass to get all weight + bias and activation levels
    let mut weights_bias_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());
    let mut activation_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());

    let activation_pass = input_array.clone();
    for layer in model.iter() {
        let (weight_pass, activation_pass) = layer.pass(activation_pass.clone());

        weights_bias_vec.push(weight_pass);
        activation_vec.push(activation_pass.clone());
    }

    // Backwards pass to get how much we should change each weight and bias
    let mut cost_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());
    let mut change_weights_bias_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());
    let mut change_activation_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());


    // Calculate init cost
    let error: Array1<f64> = &correct_answer_array - activation_vec.last().unwrap();

    // Reverse so that we dont have to go backwards in the for loop and keep doing len - idx
    // Doing this after first cost so that the first layer can be popped
    weights_bias_vec.reverse();
    activation_vec.reverse();

    for (weights_bias_value, activation_value) in zip(weights_bias_vec, activation_vec) {

    }

}