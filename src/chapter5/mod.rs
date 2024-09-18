use std::f64::MAX;

use crate::{Activity, Chapter};
use itertools::izip;
const ACTIVITIES: [Activity; 2] = [CHAPTER5A, CHAPTER5B];

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Learning multiple weights at a time",
    id: "5",
};

const CHAPTER5A: Activity = Activity {
    task: chapter5a,
    name: "Gradient descent learning with multiple weights",
    id: "5a",
};

const CHAPTER5B: Activity = Activity {
    task: chapter5b,
    name: "Freezing one weight",
    id: "5b",
};

fn dot(i: &Vec<f64>, j: &Vec<f64>) -> f64 {
    if i.len() != j.len() {
        panic!("input vectors must be of same length")
    }
    i.into_iter()
        .zip(j.into_iter())
        .fold(0.0, |acc, (a, b)| acc + a * b)
}

fn add(i: &mut Vec<f64>, j: &Vec<f64>) -> () {
    if i.len() != j.len() {
        panic!("input vectors must be of same length")
    }
    for (idx, el) in i.iter_mut().enumerate() {
        *el +=  j[idx];
    }
}

fn ele_mul(s: f64, v: &Vec<f64>) -> Vec<f64> {
    v.iter().map(|e| e*s).collect()
}

fn chapter5a() -> Result<(), std::io::Error> {

    struct Output {
        weights: Vec<f64>,
        prediction: f64,
        iterations: i16,
        error: f64
    }

    const ALPHA: f64 = 0.01;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i16 = 2000;

    fn solve_weights(input: &Vec<f64>, weights: &Vec<f64>, true_value: f64) -> Output {
        let mut weights = weights.to_owned();

        let mut error = MAX;
        let mut pred = MAX;

        for iteration in  0..MAX_ITERATIONS {

            pred = dot(input, &weights);
            let delta = pred - true_value;
            error = delta * delta;

            if error < TOLERANCE {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error
                };
            }

            let weight_deltas = ele_mul(-1.0*delta*ALPHA, input);
            add(&mut weights, &weight_deltas);

        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error
        };

    }
    let win_or_lose_binary: Vec<f64> = vec![1.0,1.0,0.0,1.0];
    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    let weights = vec![0.1, 0.2, -0.1];

    for (t, w, n, wlb) in izip!(toes, wlrec, nfans, win_or_lose_binary) {
        let input = vec![t, w, n];
        let res = solve_weights(&input, &weights, wlb);
        println!("Input: {:?}, True value: {:.2}, Prediction: {:.2}, Error: {:.2}, Weights: {:?}, Iterations: {}", input, wlb, res.prediction, res.error, res.weights, res.iterations);
    }
    Ok(())

}

fn chapter5b() -> Result<(), std::io::Error> {

    struct Output {
        weights: Vec<f64>,
        prediction: f64,
        iterations: i16,
        error: f64
    }

    const ALPHA: f64 = 0.3;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i16 = 2000;

    fn solve_weights(input: &Vec<f64>, weights: &Vec<f64>, true_value: f64) -> Output {
        let mut weights = weights.to_owned();

        let mut error = MAX;
        let mut pred = MAX;

        for iteration in  0..MAX_ITERATIONS {

            pred = dot(input, &weights);
            let delta = pred - true_value;
            error = delta * delta;

            if error < TOLERANCE {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error
                };
            }

            let mut weight_deltas = ele_mul(-1.0*delta*ALPHA, input);
            weight_deltas[0] = 0.0;
            add(&mut weights, &weight_deltas);

        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error
        };

    }
    let win_or_lose_binary: Vec<f64> = vec![1.0,1.0,0.0,1.0];
    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    let weights = vec![0.1, 0.2, -0.1];

    for (t, w, n, wlb) in izip!(toes, wlrec, nfans, win_or_lose_binary) {
        let input = vec![t, w, n];
        let res = solve_weights(&input, &weights, wlb);
        println!("Input: {:?}, True value: {:.2}, Prediction: {:.2}, Error: {:.2}, Weights: {:?}, Iterations: {}", input, wlb, res.prediction, res.error, res.weights, res.iterations);
    }
    Ok(())

}