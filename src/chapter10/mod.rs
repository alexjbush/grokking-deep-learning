use std::time::{Duration, Instant};

use crate::{Activity, Chapter};
use ndarray::{s, Array, Array2, ArrayBase, Axis, Dim, Dimension, OwnedRepr};
const ACTIVITIES: [Activity; 1] = [CHAPTER10A];
use mnist::*;
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Distribution, Uniform},
    RandomExt,
};
use ndarray_stats::QuantileExt;
use rand_chacha::ChaCha8Rng;

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Neural learning about edges and corners",
    id: "10",
};

const CHAPTER10A: Activity = Activity {
    task: chapter10a,
    name: "A simple implementation in NDARRAY",
    id: "10a",
};

fn tanh(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| return v.tanh())
}

fn tanh2deriv(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| return 1.0 - v.powi(2))
}

fn softmax(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    let temp = m.exp();
    let sum = temp.sum_axis(Axis(1)).insert_axis(Axis(1));
    return temp / sum;
}

fn get_image_section(
    layer: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    row_from: usize,
    row_to: usize,
    col_from: usize,
    col_to: usize,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 4]>> {
    let section = layer.slice(s![.., row_from..row_to, col_from..col_to]);
    let dim3 = col_to - col_from;
    let dim2 = row_to - row_from;
    let dim1 = 1;
    let dim0 = section.raw_dim().size() / (dim3 * dim2 * dim1);
    return section.to_shape((dim0, 1, row_to - row_from, col_to - col_from)).unwrap().try_into_owned_nocopy().unwrap();
}

fn chapter10a() -> Result<(), std::io::Error> {
    let alpha = 2.0;
    let max_iterations = 300;
    let hidden_size = 100;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;
    let batch_size = 128;

    let input_rows = 28;
    let input_cols = 28;

    let kernel_rows = 3;
    let kernel_cols = 3;
    let num_kernels = 16;

    let hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .training_set_length(train_image_count as u32)
        .test_set_length(test_image_count as u32)
        .base_path(".data/")
        .base_url("https://azureopendatastorage.blob.core.windows.net/mnist")
        .download_and_extract()
        .label_format_one_hot()
        .finalize();

    let images = Array2::from_shape_vec((train_image_count, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 255.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 255.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut kernels = Array::random_using(
        (kernel_rows * kernel_cols, num_kernels),
        Uniform::new(-0.01, 0.01),
        &mut rng,
    );

    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    let mut total_duration = Duration::new(0, 0);

    for iteration in 0..max_iterations {
        let start = Instant::now();
        let mut correct_cnt: usize = 0;

        for i in 0..(train_image_count / batch_size) {
            let batch_start = i * batch_size;
            let batch_end = (i + 1) * batch_size;
            let _layer_0 = images.slice(s![batch_start..batch_end, ..]);
            let layer_0 = _layer_0.to_shape((_layer_0.dim().0, 20, 28)).unwrap();
            //here
            let mut layer_1 = tanh(&layer_0.dot(&weights_0_1));
            let mask = Array::random_using(
                layer_1.raw_dim(),
                Uniform::new(0, 2).map(|e| e as f64),
                &mut rng,
            );
            layer_1 = layer_1 * &mask * 2.0;
            let layer_2 = softmax(&layer_1.dot(&weights_1_2));

            let layer_2_delta = (&layer_2 - &labels.slice(s![batch_start..batch_end, ..]))
                / (batch_size as f64 * layer_2.dim().0 as f64);
            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * tanh2deriv(&layer_1) * &mask;

            weights_1_2 = weights_1_2 - (alpha) * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - (alpha) * (layer_0.t().dot(&layer_1_delta));
            for k in 0..batch_size {
                let correct = &layer_2.row(k).argmax().unwrap()
                    == &labels.row(batch_start + k).argmax().unwrap();
                if correct {
                    correct_cnt += 1;
                }
            }
        }
        let duration = start.elapsed();
        total_duration += duration;

        if iteration % 10 == 0 || iteration == max_iterations - 1 {
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = tanh(&layer_0.dot(&weights_0_1));
                let layer_2 = layer_1.dot(&weights_1_2);

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Correct: {:.3}, Test-Correct: {:.3}",
                iteration,
                correct_cnt as f64 / train_image_count as f64,
                test_correct_cnt as f64 / test_image_count as f64
            );
        }
    }

    println!(
        "Average iteration duration: {:?}",
        total_duration / max_iterations
    );

    Ok(())
}
