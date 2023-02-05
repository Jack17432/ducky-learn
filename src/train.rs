use std::iter::zip;
use ndarray::{arr1, Array1, Array2};
use super::layers::*;

pub fn train<L>(model: &Vec<L>,
                train_data: Array2<f64>, train_lbl: Array2<f64>,
                test_data: Array2<f64>, test_lbl: Array2<f64>)
where
    L: Layer1d
{
    todo!()
}

//noinspection RsBorrowChecker For some reason it says that the item is moved eventhough it isn't
pub fn forward_pass<L>(model: &Vec<L>, data: Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>)
where
    L: Layer1d
{
    let mut weights_bias_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());
    let mut activation_vec: Vec<Array1<f64>> = Vec::with_capacity(model.len());

    let mut activation_pass = data;
    let mut weight_pass;

    for layer in model.iter() {
        (weight_pass, activation_pass) = layer.pass(activation_pass.clone());

        weights_bias_vec.push(weight_pass);
        activation_vec.push(activation_pass.clone());
    }

    (weights_bias_vec, activation_vec)
}
