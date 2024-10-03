use crate::{Activity, Chapter};
use ndarray::{array, s, Array, Array2, ArrayBase, Dim, OwnedRepr};
const ACTIVITIES: [Activity; 1] = [CHAPTER8A];

use mnist::*;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
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

fn relu(m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| {
        if *v > 0.0 {
            return *v
        } else {
            return 0.0
        }
    })
}

fn relu2deriv(m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> {
    m.map(|v| {
        if *v > 0.0 {
            return 1.0
        } else {
            return 0.0
        }
    })
}

fn chapter8a() -> Result<(), std::io::Error> {
    let alpha = 0.005;
    let tolerance = 0.000000001;
    let max_iterations = 2000;
    let hidden_size = 40;
    let pixels_per_image = 784;
    let num_labels = 10;
   
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new().base_path(".data/").base_url("https://azureopendatastorage.blob.core.windows.net/mnist").download_and_extract().label_format_one_hot().finalize();

    let train_images = Array2::from_shape_vec((50_000, 28 * 28), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let train_labels = Array2::from_shape_vec((50_000, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let test_images = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let test_labels = Array2::from_shape_vec((10_000, 10), tst_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let mut complete = false;
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut weights_0_1 = Array::random_using((pixels_per_image, hidden_size), Uniform::new(-0.1,0.1), &mut rng);
    let mut weights_1_2 = Array::random_using((hidden_size, num_labels), Uniform::new(-0.1,0.1), &mut rng);
  

    for iteration in 0..max_iterations {
        let mut error = 0.0;
        let mut correct_cnt = 0;

        if error < tolerance {
            println!(
                "Iterations: {}, Error: {}, Correct: {:?}",
                iteration, error, correct_cnt
            );
            complete = true;
            break;
        }
    }

    if !complete {
        println!("Failed to find solution under tolerance.")
    }

    Ok(())
}
