#![allow(dead_code)]
#![doc(html_logo_url = "https://img.freepik.com/free-icon/rubber-duck_318-763202.jpg?w=2000")]


extern crate ndarray;

pub mod layers;

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


