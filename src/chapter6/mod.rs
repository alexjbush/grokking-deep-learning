use crate::{Activity, Chapter};
use ndarray::{array, s, Array, ArrayBase, Dim, OwnedRepr};
const ACTIVITIES: [Activity; 3] = [CHAPTER6A, CHAPTER6B, CHAPTER6C];
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_chacha::ChaCha8Rng;

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Building your first deep neural network",
    id: "6",
};

const CHAPTER6A: Activity = Activity {
    task: chapter6a,
    name: "Learning the whole dataset",
    id: "6a",
};

fn chapter6a() -> Result<(), std::io::Error> {
    let mut weights = array![0.5, 0.48, -0.7];
    let alpha = 0.1;

    let tolerance = 0.000000001;
    let max_iterations = 2000;

    let street_lights = array![
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ];

    let walk_vs_stop = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let mut complete = false;

    for iteration in 0..max_iterations {
        let mut error_for_all_lights = 0.0;
        for row_index in 0..walk_vs_stop.len() {
            let input = street_lights.row(row_index);
            let goal_prediction = walk_vs_stop[row_index];

            let prediction = input.dot(&weights);

            let delta = prediction - goal_prediction;
            let error = delta * delta;
            error_for_all_lights += error;

            weights = weights - ((alpha * delta) * &input);
        }
        if error_for_all_lights < tolerance {
            println!(
                "Iterations: {}, Error: {}, Weights: {:?}",
                iteration, error_for_all_lights, weights
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

const CHAPTER6B: Activity = Activity {
    task: chapter6b,
    name: "Learning the whole dataset with full GD",
    id: "6b",
};

fn chapter6b() -> Result<(), std::io::Error> {
    let mut weights = array![0.5, 0.48, -0.7];
    let alpha = 0.1;

    let tolerance = 0.000000001;
    let max_iterations = 20000;

    let street_lights = array![
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ];

    let walk_vs_stop = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let mut complete = false;

    for iteration in 0..max_iterations {
        let mut error_for_all_lights = 0.0;
        let mut weight_delta_for_all_lights: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> =
            Array::zeros(street_lights.len_of(ndarray::Axis(1)));

        for row_index in 0..walk_vs_stop.len() {
            let input = street_lights.row(row_index);
            let goal_prediction = walk_vs_stop[row_index];

            let prediction = input.dot(&weights);

            let delta = prediction - goal_prediction;
            let error = delta * delta;

            error_for_all_lights += error;
            let weight_delta = (alpha * delta) * &input;
            weight_delta_for_all_lights = weight_delta_for_all_lights + weight_delta;
        }
        weight_delta_for_all_lights = weight_delta_for_all_lights / walk_vs_stop.len() as f64;

        weights = weights - weight_delta_for_all_lights;

        if error_for_all_lights < tolerance {
            println!(
                "Iterations: {}, Error: {}, Weights: {:?}",
                iteration, error_for_all_lights, weights
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

const CHAPTER6C: Activity = Activity {
    task: chapter6c,
    name: "Putting it all together",
    id: "6c",
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

fn chapter6c() -> Result<(), std::io::Error> {
    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let streetlights = array![
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];

    let walk_vs_stop = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0].t().to_owned();

    const ALPHA: f64 = 0.2;
    const HIDDEN_SIZE: usize = 4;

    let tolerance = 0.000000001;
    let max_iterations = 20000;

    let mut weights_0_1 = Array::random_using((3, HIDDEN_SIZE), Uniform::new(-1.0,1.0), &mut rng);
    let mut weights_1_2 = Array::random_using((HIDDEN_SIZE, 1), Uniform::new(-1.0,1.0), &mut rng);
    
    let mut complete = false;

    for iteration in 0..max_iterations {
        let mut layer_2_error = 0.0;

        for i in 0..streetlights.dim().0 {
            let layer_0: ArrayBase<ndarray::ViewRepr<&f64>, Dim<[usize; 2]>> = streetlights.slice(s![i..i+1,..]);
            let layer_1 = relu(&layer_0.dot(&weights_0_1));
            let layer_2 = layer_1.dot(&weights_1_2);

            let layer_2_delta: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = layer_2 - walk_vs_stop[i];
            layer_2_error += layer_2_delta.map(|v| *v * *v).sum();

            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t()) * relu2deriv(&layer_1);

            weights_1_2 = weights_1_2 - ALPHA * (layer_1.t().dot(&layer_2_delta));
            weights_0_1 = weights_0_1 - ALPHA * (layer_0.t().dot(&layer_1_delta));

        }

        if iteration % 10 == 9 {
            println!("Error: {}", layer_2_error)
        }

        if layer_2_error < tolerance {
            println!(
                "Iterations: {}, Error: {:?}, Weights 0->1: {:?}, Weights 1->2: {:?}",
                iteration, layer_2_error, weights_0_1, weights_1_2
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
