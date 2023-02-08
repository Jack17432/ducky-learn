// use ducky_learn::layers::*;
use ducky_learn::util::one_hot_encoding_vec;
use mnist::*;
use ndarray::prelude::*;

fn main() {
    let (_train_data, _train_labels, _test_data, _test_labels, _val_data, _val_labels) =
        create_mnist_dataset(50_000, 10_000, 10_000);
}

fn create_mnist_dataset(
    trn_len: u32,
    tst_len: u32,
    val_len: u32,
) -> (
    Array2<f64>,
    Array2<f64>, // Training data, label
    Array2<f64>,
    Array2<f64>, // Testing data, label
    Array2<f64>,
    Array2<f64>, // Validation data, label
) {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        val_img,
        val_lbl,
    } = MnistBuilder::new()
        .training_set_length(trn_len)
        .test_set_length(tst_len)
        .validation_set_length(val_len)
        .finalize();

    let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let trn_lbl: Vec<usize> = trn_lbl.iter().map(|x| *x as usize).collect();
    let train_labels: Array2<f64> = one_hot_encoding_vec(&trn_lbl, 9);

    let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let tst_lbl: Vec<usize> = tst_lbl.iter().map(|x| *x as usize).collect();
    let test_labels: Array2<f64> = one_hot_encoding_vec(&tst_lbl, 9);

    let val_data = Array2::from_shape_vec((10_000, 784), val_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.);

    let val_lbl: Vec<usize> = val_lbl.iter().map(|x| *x as usize).collect();
    let val_labels: Array2<f64> = one_hot_encoding_vec(&val_lbl, 9);

    (
        train_data,
        train_labels,
        test_data,
        test_labels,
        val_data,
        val_labels,
    )
}
