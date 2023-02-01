// use ducky_learn::layers::*;
use mnist::*;
use ndarray::prelude::*;

fn main() {
     let (
         _train_data, _train_labels,
         _test_data, _test_labels,
         _val_data, _val_labels
     ) = create_mnist_dataset(50_000, 10_000, 10_000);
}

fn create_mnist_dataset(trn_len: u32, tst_len: u32, val_len: u32) -> (
    Array3<f32>, Array2<f32>,   // Training data, label
    Array3<f32>, Array2<f32>,   // Testing data, label
    Array3<f32>, Array2<f32>    // Validation data, label
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

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    let val_data = Array3::from_shape_vec((10_000, 28, 28), val_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    let val_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), val_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, test_data, test_labels, val_data, val_labels)
}
