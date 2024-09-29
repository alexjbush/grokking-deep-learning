use crate::{Activity, Chapter};
use ndarray::{array, Array, ArrayBase, Dim, OwnedRepr};
const ACTIVITIES: [Activity; 2] = [CHAPTER6A, CHAPTER6B];

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
