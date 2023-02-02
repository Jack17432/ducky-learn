#![allow(dead_code)]
#![doc(html_logo_url = "https://img.freepik.com/free-icon/rubber-duck_318-763202.jpg?w=2000")]


extern crate ndarray;

pub mod layers;
pub mod util;
pub mod activations;
// TODO: Create a training mod
// TODO: Create a back prop mod

#[cfg(test)]
mod layers_tests {
    use ndarray::*;
    use super::layers::*;

    #[test]
    fn dense1d_pass_arr1_1() {
        let layer = Dense1d::from(
            |x| x,
            arr2(&[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
            arr1(&[1., 1., 1.]),
        );
        let input_array = arr1(&[1., 1., 1.]);

        assert_eq!(
            layer.pass(input_array),
            arr1(&[4., 4., 4.])
        )
    }

    #[test]
    fn dense1d_pass_arr1_2() {
        let layer = Dense1d::from(
            |x| x,
            arr2(&[
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
            ]),
            arr1(&[
                1., 1., 1.
            ]),
        );
        let input_array = arr1(&[
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ]);

        assert_eq!(
            layer.pass(input_array),
            arr1(&[13.0, 13.0, 13.0])
        )
    }

    #[test]
    #[should_panic]
    fn dense1d_pass_arr1_diff_size() {
        let layer = Dense1d::from(
            |x| x,
            arr2(&[[1., 1., 1., 1.], [1., 1., 1., 1.]]),
            arr1(&[0., 0.]),
        );
        let input_array = arr1(&[1.]);

        layer.pass(input_array);
    }

    #[test]
    fn dense1d_new() {
        let layer = Dense1d::new(5, 10, |x| x);

        let input_array = arr1(&[
            1., 1., 1., 1., 1.
        ]);

        layer.pass(input_array);
    }
}

#[cfg(test)]
mod util_tests {
    use ndarray::*;
    use super::util::*;

    #[test]
    fn one_hot_encoding_vec_std_use() {
        let input_array = vec![3, 1, 0];
        let output_test_array = arr2(
            &[[0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]]
        );

        assert_eq!(one_hot_encoding_vec(&input_array, 3),
                   output_test_array);
    }

    #[test]
    fn one_hot_encoding_vec_zero_input() {
        let input_array = vec![];
        let output_test_array = Array2::zeros((0, 0));

        assert_eq!(one_hot_encoding_vec(&input_array, 0),
                   output_test_array);
    }
}

#[cfg(test)]
mod activations_tests {
    use super::activations::*;
    use ndarray::arr1;

    #[test]
    fn relu_1d_1() {
        let input_array = arr1(&[0., 1., -1., 0.01, -0.1]);

        assert_eq!(relu_1d(input_array),
                   arr1(&[0., 1., 0., 0.01, 0.]));
    }

    #[test]
    fn relu_1d_2() {
        let input_array = arr1(&[]);

        assert_eq!(relu_1d(input_array),
                   arr1(&[]));
    }

    #[test]
    fn relu_1d_3() {
        let input_array = arr1(&[-1.3456435325242, -32145324321., -132432888.]);

        assert_eq!(relu_1d(input_array),
                   arr1(&[0., 0., 0.]));
    }

    #[test]
    fn softmax_1d_1() {
        let input_array = arr1(&[0., 1., -1., 0.01, -0.1]);

        assert_eq!(softmax_1d(input_array),
                   arr1(&[0.16663753690463112, 0.4529677885070323, 0.0613025239546613, 0.16831227199301688, 0.15077987864065834]));
    }

    #[test]
    fn softmax_1d_2() {
        let input_array = arr1(&[]);

        assert_eq!(softmax_1d(input_array),
                   arr1(&[]));
    }

    #[test]
    fn softmax_1d_3() {
        let input_array = arr1(&[-0.3456435325242, 232., -888.]);

        assert_eq!(softmax_1d(input_array),
                   arr1(&[1.2404210269803915e-101, 1.0, 0.0]));
    }
}