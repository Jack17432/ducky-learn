#![allow(dead_code)]
#![doc(html_logo_url = "https://img.freepik.com/free-icon/rubber-duck_318-763202.jpg?w=2000")]


extern crate ndarray;

pub mod layers;
pub mod util;

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
            arr1(&[
                13.0, 13.0, 13.0
            ])
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
}

#[cfg(test)]
mod util_tests {
    use ndarray::*;
    use super::util::*;

    #[test]
    fn one_hot_encoding_vec_std_use() {
        let input_array = vec![3., 1., 0.];
        let output_test_array = arr2(
            &[[0., 0., 0., 1.], [0., 1., 0., 0.], [1., 0., 0., 0.]]
        );

        assert_eq!(one_hot_encoding_vec(&input_array, 3), output_test_array);
    }

    #[test]
    fn one_hot_encoding_vec_zero_input() {
        let input_array = vec![];
        let output_test_array = Array2::zeros((0, 0));

        assert_eq!(one_hot_encoding_vec(&input_array, 0), output_test_array);
    }
}

