use super::layers::*;
use ndarray::{arr1, Array1, Array2};
use std::iter::zip;

pub fn train<L>(
    model: &Vec<L>,
    train_data: Array2<f64>,
    train_lbl: Array2<f64>,
    test_data: Array2<f64>,
    test_lbl: Array2<f64>,
) where
    L: Layer1d,
{
    todo!()
}

//noinspection RsBorrowChecker For some reason it says that the item is moved eventhough it isn't
pub fn forward_pass<L>(model: &Vec<L>, data: Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array1<f64>>)
where
    L: Layer1d,
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

pub fn back_propagation<L>(
    model: &Vec<L>,
    weights_bias_vec: Vec<Array1<f64>>,
    activation_vec: Vec<Array1<f64>>,
    target_out: Array1<f64>,
) where
    L: Layer1d,
{
    todo!()
}

#[cfg(test)]
mod train_tests {
    use super::*;
    use crate::layers::*;
    use crate::activations::*;
    use ndarray::arr1;

    #[test]
    fn forwards_pass_1() {
        let model = vec![
            Dense1d::new(1, 3, relu_1d, deriv_relu_1d),
            Dense1d::new(3, 5, relu_1d, deriv_relu_1d),
            Dense1d::new(5, 10, softmax_1d, deriv_relu_1d),
        ];

        let (weights_bias_vec, activation_vec) = forward_pass(&model, arr1(&[1.]));

        assert_eq!(weights_bias_vec.len(), 3);
        assert_eq!(activation_vec.len(), 3)
    }

    #[test]
    fn forwards_pass_2() {
        let model = vec![
            Dense1d::new(5, 5, relu_1d, deriv_relu_1d),
            Dense1d::new(5, 5, relu_1d, deriv_relu_1d),
            Dense1d::new(5, 5, softmax_1d, deriv_relu_1d),
        ];

        let (weights_bias_vec, activation_vec) =
            forward_pass(&model, arr1(&[1., 2., 0.2, 1., 0.32]));

        assert_eq!(weights_bias_vec.first().unwrap().shape(), [5]);
        assert_eq!(activation_vec.first().unwrap().shape(), [5])
    }

    #[test]
    fn forwards_pass_3() {
        let model = vec![
            Dense1d::new(5, 3, relu_1d, deriv_relu_1d),
            Dense1d::new(3, 5, relu_1d, deriv_relu_1d),
            Dense1d::new(5, 10, softmax_1d, deriv_relu_1d),
        ];

        let (weights_bias_vec, activation_vec) =
            forward_pass(&model, arr1(&[1., 2., 0.2, 1., 0.32]));

        assert_eq!(weights_bias_vec.last().unwrap().shape(), [10]);
        assert_eq!(activation_vec.last().unwrap().shape(), [10])
    }

    #[test]
    #[should_panic]
    fn forwards_pass_4() {
        let model = vec![
            Dense1d::new(5, 3, relu_1d, deriv_relu_1d),
            Dense1d::new(4, 5, relu_1d, deriv_relu_1d),
            Dense1d::new(5, 10, softmax_1d, deriv_relu_1d),
        ];

        let (weights_bias_vec, activation_vec) =
            forward_pass(&model, arr1(&[1., 2., 0.2, 1., 0.32]));
    }
}
