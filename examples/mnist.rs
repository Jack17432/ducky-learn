// use ducky_learn::layers::*;
use mnist::*;
use ndarray::prelude::*;
use ducky_learn::util::one_hot_encoding_vec;

fn main() {
     let (
         _train_data, _train_labels,
         _test_data, _test_labels,
         _val_data, _val_labels
     ) = create_mnist_dataset(50_000, 10_000, 10_000);
}

fn create_mnist_dataset(trn_len: u32, tst_len: u32, val_len: u32) -> (
    Array3<f64>, Array2<f64>,   // Training data, label
    Array3<f64>, Array2<f64>,   // Testing data, label
    Array3<f64>, Array2<f64>    // Validation data, label
){
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


    // TODO: change this to 50_000 by 784 so that its 1d data
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    // Convert the returned Mnist struct to Array2 format
    let trn_lbl: Vec<f64> = trn_lbl.iter().map(|x| *x as f64).collect();
    let train_labels: Array2<f64> = one_hot_encoding_vec(&trn_lbl, 9);

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let tst_lbl: Vec<f64> = tst_lbl.iter().map(|x| *x as f64).collect();
    let test_labels: Array2<f64> = one_hot_encoding_vec(&tst_lbl, 9);

    let val_data = Array3::from_shape_vec((10_000, 28, 28), val_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let val_lbl: Vec<f64> = val_lbl.iter().map(|x| *x as f64).collect();
    let val_labels: Array2<f64> = one_hot_encoding_vec(&val_lbl, 9);

    (train_data, train_labels, test_data, test_labels, val_data, val_labels)
}
