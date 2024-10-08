use std::time::{Duration, Instant};

use crate::{Activity, Chapter};
use ndarray::{array, s, Array, Array2, ArrayBase, Dim, OwnedRepr};
const ACTIVITIES: [Activity; 6] = [
    CHAPTER8A, CHAPTER8B, CHAPTER8C, CHAPTER8D, CHAPTER8E, CHAPTER8F,
];
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
    name: "Learning signal and ignoring noise",
    id: "8",
};

const CHAPTER8A: Activity = Activity {
    task: chapter8a,
    name: "Three-layer network on MNIST",
    id: "8a",
};

const CHAPTER8B: Activity = Activity {
    task: chapter8b,
    name: "Well, that was easy",
    id: "8b",
};

const CHAPTER8C: Activity = Activity {
    task: chapter8c,
    name: "Early Stopping",
    id: "8c",
};

const CHAPTER8D: Activity = Activity {
    task: chapter8d,
    name: "Drop out",
    id: "8d",
};

const CHAPTER8E: Activity = Activity {
    task: chapter8e,
    name: "Batch Gradient Descent",
    id: "8e",
};

const CHAPTER8F: Activity = Activity {
    task: chapter8f,
    name: "Batch Gradient Descent (fast)",
    id: "8f",
};

fn relu(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| if *v > 0.0 { return *v } else { return 0.0 })
}

fn relu2deriv(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| if *v > 0.0 { return 1.0 } else { return 0.0 })
}

fn chapter8a() -> Result<(), std::io::Error> {
    let alpha = 0.005;
    let tolerance = 0.000000001;
    let max_iterations = 350;
    let hidden_size = 40;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;

    let Mnist {
        trn_img, trn_lbl, ..
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
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut complete = false;
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    for iteration in 0..max_iterations {
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..train_image_count {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                images.slice(s![i..i + 1, ..]);
            let layer_1 = relu(&layer_0.dot(&weights_0_1));
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                &layer_2 - &labels.slice(s![i..i + 1, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let correct =
                layer_2.argmax().unwrap().1 == labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
            if correct {
                correct_cnt += 1;
            }

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1);

            weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
        }

        println!(
            "Iterations: {}, Error: {}, Correct: {:.2}",
            iteration,
            layer_2_error as f64 / train_image_count as f64,
            correct_cnt as f64 / train_image_count as f64
        );

        if layer_2_error < tolerance {
            complete = true;
            break;
        }
    }

    if !complete {
        println!("Failed to find solution under tolerance.")
    }

    Ok(())
}

fn chapter8b() -> Result<(), std::io::Error> {
    let alpha = 0.005;
    let tolerance = 0.000000001;
    let max_iterations = 350;
    let hidden_size = 40;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;

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
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut complete = false;
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    for iteration in 0..max_iterations {
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..train_image_count {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                images.slice(s![i..i + 1, ..]);
            let layer_1 = relu(&layer_0.dot(&weights_0_1));
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                &layer_2 - &labels.slice(s![i..i + 1, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let correct =
                layer_2.argmax().unwrap().1 == labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
            if correct {
                correct_cnt += 1;
            }

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1);

            weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
        }

        if iteration % 10 == 0 || iteration == max_iterations - 1 {
            let mut test_layer_2_error = 0.0;
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = relu(&layer_0.dot(&weights_0_1));
                let layer_2 = layer_1.dot(&weights_1_2);

                let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    &layer_2 - &test_labels.slice(s![i..i + 1, ..]);
                test_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Error: {:.3}, Train-Correct: {:.3}, Test-Error: {:.3}, Test-Correct: {:.3}",
                iteration,
                layer_2_error as f64 / train_image_count as f64,
                correct_cnt as f64 / train_image_count as f64,
                test_layer_2_error as f64 / test_image_count as f64,
                test_correct_cnt as f64 / test_image_count as f64
            );
        }

        if layer_2_error < tolerance {
            complete = true;
            break;
        }
    }

    if !complete {
        println!("Failed to find solution under tolerance.")
    }

    Ok(())
}

fn chapter8c() -> Result<(), std::io::Error> {
    let alpha = 0.005;
    let max_iterations = 350;
    let hidden_size = 40;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;
    let validation_image_count: usize = 1_000;
    let min_num_decline: i32 = 3;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        val_img,
        val_lbl,
        ..
    } = MnistBuilder::new()
        .training_set_length(train_image_count as u32)
        .test_set_length(test_image_count as u32)
        .validation_set_length(validation_image_count as u32)
        .base_path(".data/")
        .base_url("https://azureopendatastorage.blob.core.windows.net/mnist")
        .download_and_extract()
        .label_format_one_hot()
        .finalize();

    let images = Array2::from_shape_vec((train_image_count, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let validation_images = Array2::from_shape_vec((validation_image_count, 28 * 28), val_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let validation_labels = Array2::from_shape_vec((validation_image_count, 10), val_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    struct Best {
        iteration: i32,
        weights_0_1: Array2<f64>,
        weights_1_2: Array2<f64>,
        layer_2_error: f64,
        correct_cnt: usize,
        validation_layer_2_error: f64,
        validation_correct_cnt: usize,
    }

    let mut best = Best {
        iteration: 0,
        weights_0_1: array![[]],
        weights_1_2: array![[]],
        layer_2_error: f64::MAX,
        correct_cnt: usize::MAX,
        validation_layer_2_error: f64::MAX,
        validation_correct_cnt: usize::MAX,
    };

    for iteration in 0..max_iterations {
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..train_image_count {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                images.slice(s![i..i + 1, ..]);
            let layer_1 = relu(&layer_0.dot(&weights_0_1));
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                &layer_2 - &labels.slice(s![i..i + 1, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let correct =
                layer_2.argmax().unwrap().1 == labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
            if correct {
                correct_cnt += 1;
            }

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1);

            weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
        }

        let mut validation_layer_2_error = 0.0;
        let mut validation_correct_cnt: usize = 0;

        for i in 0..validation_image_count {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                validation_images.slice(s![i..i + 1, ..]);
            let layer_1 = relu(&layer_0.dot(&weights_0_1));
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                &layer_2 - &validation_labels.slice(s![i..i + 1, ..]);
            validation_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let correct = layer_2.argmax().unwrap().1
                == validation_labels
                    .slice(s![i..i + 1, ..])
                    .argmax()
                    .unwrap()
                    .1;
            if correct {
                validation_correct_cnt += 1;
            }
        }

        if validation_layer_2_error < best.validation_layer_2_error {
            best = Best {
                iteration: iteration,
                weights_0_1: weights_0_1.clone(),
                weights_1_2: weights_1_2.clone(),
                layer_2_error: layer_2_error,
                correct_cnt: correct_cnt,
                validation_layer_2_error: validation_layer_2_error,
                validation_correct_cnt: validation_correct_cnt,
            };
        }

        if iteration - best.iteration > min_num_decline {
            let mut test_layer_2_error = 0.0;
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = relu(&layer_0.dot(&best.weights_0_1));
                let layer_2 = layer_1.dot(&best.weights_1_2);

                let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    &layer_2 - &test_labels.slice(s![i..i + 1, ..]);
                test_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Error: {:.3}, Train-Correct: {:.3}, Validation-Error: {:.3}, Validation-Correct: {:.3}, Test-Error: {:.3}, Test-Correct: {:.3}",
                best.iteration,
                best.layer_2_error as f64 / train_image_count as f64,
                best.correct_cnt as f64 / train_image_count as f64,
                best.validation_layer_2_error as f64 / validation_image_count as f64,
                best.validation_correct_cnt as f64 / validation_image_count as f64,
                test_layer_2_error as f64 / test_image_count as f64,
                test_correct_cnt as f64 / test_image_count as f64
            );
            break;
        }
    }

    Ok(())
}

fn chapter8d() -> Result<(), std::io::Error> {
    let alpha = 0.005;
    let max_iterations = 350;
    let hidden_size = 40;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;

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
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    let mut total_duration = Duration::new(0, 0);

    for iteration in 0..max_iterations {
        let start = Instant::now();
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..train_image_count {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                images.slice(s![i..i + 1, ..]);
            let mut _layer_1 = relu(&layer_0.dot(&weights_0_1));
            let mask = Array::random_using(
                _layer_1.raw_dim(),
                Uniform::new(0, 2).map(|e| e as f64),
                &mut rng,
            );
            let layer_1 = _layer_1 * &mask * 2.0;
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                &layer_2 - &labels.slice(s![i..i + 1, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let correct =
                layer_2.argmax().unwrap().1 == labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
            if correct {
                correct_cnt += 1;
            }

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1) * mask;

            weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
        }

        let duration: std::time::Duration = start.elapsed();
        total_duration += duration;

        if iteration % 10 == 0 || iteration == max_iterations - 1 {
            let mut test_layer_2_error = 0.0;
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = relu(&layer_0.dot(&weights_0_1));
                let layer_2 = layer_1.dot(&weights_1_2);

                let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    &layer_2 - &test_labels.slice(s![i..i + 1, ..]);
                test_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Error: {:.3}, Train-Correct: {:.3}, Test-Error: {:.3}, Test-Correct: {:.3}",
                iteration,
                layer_2_error as f64 / train_image_count as f64,
                correct_cnt as f64 / train_image_count as f64,
                test_layer_2_error as f64 / test_image_count as f64,
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

fn chapter8e() -> Result<(), std::io::Error> {
    let alpha = 0.001;
    let max_iterations = 300;
    let hidden_size = 100;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;
    let batch_size = 100;

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
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    let mut total_duration = Duration::new(0, 0);

    for iteration in 0..max_iterations {
        let start = Instant::now();
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..(train_image_count / batch_size) {
            let batch_start = i * batch_size;
            let batch_end = (i + 1) * batch_size;
            let layer_0 = images.slice(s![batch_start..batch_end, ..]);
            let mut layer_1 = relu(&layer_0.dot(&weights_0_1));
            let mask = Array::random_using(
                layer_1.raw_dim(),
                Uniform::new(0, 2).map(|e| e as f64),
                &mut rng,
            );
            layer_1 = layer_1 * &mask * 2.0;
            let layer_2 = layer_1.dot(&weights_1_2);
            let layer_2_delta = &layer_2 - &labels.slice(s![batch_start..batch_end, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let layer_2_delta_scaled = layer_2_delta / batch_size as f64;

            for k in 0..batch_size {
                let correct = &layer_2.row(k).argmax().unwrap()
                    == &labels.row(batch_start + k).argmax().unwrap();
                if correct {
                    correct_cnt += 1;
                }

                let layer_1_delta =
                    layer_2_delta_scaled.dot(&weights_1_2.t()) * relu2deriv(&layer_1) * &mask;

                weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta_scaled));
                weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
            }
        }
        let duration = start.elapsed();
        total_duration += duration;

        if iteration % 10 == 0 || iteration == max_iterations - 1 {
            let mut test_layer_2_error = 0.0;
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = relu(&layer_0.dot(&weights_0_1));
                let layer_2 = layer_1.dot(&weights_1_2);

                let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    &layer_2 - &test_labels.slice(s![i..i + 1, ..]);
                test_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Error: {:.3}, Train-Correct: {:.3}, Test-Error: {:.3}, Test-Correct: {:.3}",
                iteration,
                layer_2_error as f64 / train_image_count as f64,
                correct_cnt as f64 / train_image_count as f64,
                test_layer_2_error as f64 / test_image_count as f64,
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

// Note this one does the sum rather than avg
fn chapter8f() -> Result<(), std::io::Error> {
    let alpha = 0.001;
    let max_iterations = 300;
    let hidden_size = 100;
    let pixels_per_image = 784;
    let num_labels = 10;
    let train_image_count: usize = 1_000;
    let test_image_count: usize = 10_000;
    let batch_size = 100;

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
        .map(|x| *x as f64 / 256.0);

    let labels = Array2::from_shape_vec((train_image_count, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((test_image_count, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((test_image_count, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using(
        (pixels_per_image, hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, num_labels), Uniform::new(-0.1, 0.1), &mut rng);

    let mut total_duration = Duration::new(0, 0);

    for iteration in 0..max_iterations {
        let start = Instant::now();
        let mut layer_2_error = 0.0;
        let mut correct_cnt: usize = 0;

        for i in 0..(train_image_count / batch_size) {
            let batch_start = i * batch_size;
            let batch_end = (i + 1) * batch_size;
            let layer_0 = images.slice(s![batch_start..batch_end, ..]);
            let mut layer_1 = relu(&layer_0.dot(&weights_0_1));
            let mask = Array::random_using(
                layer_1.raw_dim(),
                Uniform::new(0, 2).map(|e| e as f64),
                &mut rng,
            );
            layer_1 = layer_1 * &mask * 2.0;
            let layer_2 = layer_1.dot(&weights_1_2);
            let layer_2_delta = &layer_2 - &labels.slice(s![batch_start..batch_end, ..]);
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            for k in 0..batch_size {
                let correct = &layer_2.row(k).argmax().unwrap()
                    == &labels.row(batch_start + k).argmax().unwrap();
                if correct {
                    correct_cnt += 1;
                }
            }

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1) * &mask;

            weights_1_2 = weights_1_2 - alpha * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - alpha * (layer_0.t().dot(&layer_1_delta));
        }
        let duration = start.elapsed();
        total_duration += duration;

        if iteration % 10 == 0 || iteration == max_iterations - 1 {
            let mut test_layer_2_error = 0.0;
            let mut test_correct_cnt: usize = 0;

            for i in 0..test_image_count {
                let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> =
                    test_images.slice(s![i..i + 1, ..]);
                let layer_1 = relu(&layer_0.dot(&weights_0_1));
                let layer_2 = layer_1.dot(&weights_1_2);

                let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> =
                    &layer_2 - &test_labels.slice(s![i..i + 1, ..]);
                test_layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

                let correct = layer_2.argmax().unwrap().1
                    == test_labels.slice(s![i..i + 1, ..]).argmax().unwrap().1;
                if correct {
                    test_correct_cnt += 1;
                }
            }

            println!(
                "Iterations: {}, Train-Error: {:.3}, Train-Correct: {:.3}, Test-Error: {:.3}, Test-Correct: {:.3}",
                iteration,
                layer_2_error as f64 / train_image_count as f64,
                correct_cnt as f64 / train_image_count as f64,
                test_layer_2_error as f64 / test_image_count as f64,
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
