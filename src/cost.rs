use ndarray::Array1;

// TODO: Documentation
pub fn mean_squared_error(
    observed_array: Array1<f64>,
    predicted_array: Array1<f64>,
) -> Array1<f64> {
    (&observed_array - &predicted_array).mapv(|value| value.powi(2))
}

// TODO: Documentation
pub fn deriv_mean_squared_error(
    observed_array: Array1<f64>,
    predicted_array: Array1<f64>,
) -> Array1<f64> {
    -2f64 * (&observed_array - &predicted_array)
}

#[cfg(test)]
mod cost_tests {
    use super::*;
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
