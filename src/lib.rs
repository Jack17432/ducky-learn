// TODO: Create Readme.md file for project
#![allow(dead_code)]
#![allow(unused)]
#![doc(html_logo_url = "https://img.freepik.com/free-icon/rubber-duck_318-763202.jpg?w=2000")]

extern crate ndarray;

pub mod activations;
pub mod cost;
pub mod layers;
pub mod optimizers;
pub mod train;
pub mod util;

#[cfg(test)]
mod layers_tests {
    use super::activations::*;
    use super::layers::*;
    use ndarray::*;

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

#[cfg(test)]
mod util_tests {
    use super::util::*;
    use ndarray::*;

    #[test]
    fn one_hot_encoding_vec_std_use() {
        let input_array = vec![3, 1, 0];
        let output_test_array = arr2(&[[0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]]);

        assert_eq!(one_hot_encoding_vec(&input_array, 3), output_test_array);
    }

    #[test]
    fn one_hot_encoding_vec_zero_input() {
        let input_array = vec![];
        let output_test_array = Array2::zeros((0, 0));

        assert_eq!(one_hot_encoding_vec(&input_array, 0), output_test_array);
    }
}

#[cfg(test)]
mod activations_tests {
    use super::activations::*;
    use ndarray::arr1;

    #[test]
    fn relu_1d_1() {
        let input_array = arr1(&[0., 1., -1., 0.01, -0.1]);

        assert_eq!(relu_1d(input_array), arr1(&[0., 1., 0., 0.01, 0.]));
    }

    #[test]
    fn relu_1d_2() {
        let input_array = arr1(&[]);

        assert_eq!(relu_1d(input_array), arr1(&[]));
    }

    #[test]
    fn relu_1d_3() {
        let input_array = arr1(&[-1.3456435325242, -32145324321., -132432888.]);

        assert_eq!(relu_1d(input_array), arr1(&[0., 0., 0.]));
    }

    #[test]
    fn deriv_relu_1d_1() {
        let input_array = arr1(&[1.3456435325242, -32145324321., 132432888.]);
        assert_eq!(deriv_relu_1d(input_array), arr1(&[1., 0., 1.]));
    }

    #[test]
    fn deriv_relu_1d_2() {
        let input_array = arr1(&[-1.3456435325242, -32145324321., 132432888.]);
        assert_eq!(deriv_relu_1d(input_array), arr1(&[0., 0., 1.]));
    }

    #[test]
    fn deriv_relu_1d_3() {
        let input_array = arr1(&[]);
        assert_eq!(deriv_relu_1d(input_array), arr1(&[]));
    }

    #[test]
    fn softmax_1d_1() {
        let input_array = arr1(&[0., 1., -1., 0.01, -0.1]);

        assert_eq!(
            softmax_1d(input_array),
            arr1(&[
                0.16663753690463112,
                0.4529677885070323,
                0.0613025239546613,
                0.16831227199301688,
                0.15077987864065834
            ])
        );
    }

    #[test]
    fn softmax_1d_2() {
        let input_array = arr1(&[]);

        assert_eq!(softmax_1d(input_array), arr1(&[]));
    }

    #[test]
    fn softmax_1d_3() {
        let input_array = arr1(&[-0.3456435325242, 232., -888.]);

        assert_eq!(
            softmax_1d(input_array),
            arr1(&[1.2404210269803915e-101, 1.0, 0.0])
        );
    }
}

#[cfg(test)]
mod train_tests {
    use super::activations::*;
    use super::layers::*;
    use super::train::*;
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

#[cfg(test)]
mod cost_tests {
    use crate::cost::{deriv_mean_squared_error, mean_squared_error};
    use ndarray::arr1;

    #[test]
    fn mse_1() {
        let observed = arr1(&[0.88651179, 0.59085182, 0.78865531]);
        let predicted = arr1(&[0.37609094, 0.04389782, 0.27988027]);

        assert_eq!(
            mean_squared_error(observed, predicted),
            arr1(&[
                0.26052944411472256,
                0.29915867811600005,
                0.25885204132700157
            ])
        );
    }

    #[test]
    fn mse_2() {
        let observed = arr1(&[32.321, -0.32, -1.232]);
        let predicted = arr1(&[0.69953402, 0.07279993, 0.25552055]);

        assert_eq!(
            mean_squared_error(observed, predicted),
            arr1(&[999.9171107242971, 0.15429178500800492, 2.2127173866723022])
        );
    }

    #[test]
    fn mse_3() {
        let observed = arr1(&[]);
        let predicted = arr1(&[]);

        assert_eq!(mean_squared_error(observed, predicted), arr1(&[]));
    }

    #[test]
    fn deriv_mse_1() {
        let observed = arr1(&[-0.52198585, -2.27179003, -0.14017833]);
        let predicted = arr1(&[0.81674329, -1.07071564, 2.20337672]);

        assert_eq!(
            deriv_mean_squared_error(observed, predicted),
            arr1(&[2.6774582799999997, 2.40214878, 4.6871101])
        );
    }

    #[test]
    fn deriv_mse_2() {
        let observed = arr1(&[-0.76362711, -1.83292557, -0.16423367]);
        let predicted = arr1(&[-1.3829452, 0.2221366, -0.27885796]);

        assert_eq!(
            deriv_mean_squared_error(observed, predicted),
            arr1(&[-1.2386361799999999, 4.11012434, -0.22924858000000004])
        );
    }

    #[test]
    fn deriv_mse_3() {
        let observed = arr1(&[]);
        let predicted = arr1(&[]);

        assert_eq!(deriv_mean_squared_error(observed, predicted), arr1(&[]));
    }
}
