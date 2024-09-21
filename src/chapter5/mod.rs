use std::f64::MAX;

use crate::{Activity, Chapter};
use itertools::izip;
const ACTIVITIES: [Activity; 4] = [CHAPTER5A, CHAPTER5B, CHAPTER5C, CHAPTER5D];

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

const CHAPTER5C: Activity = Activity {
    task: chapter5c,
    name: "Gradient descent learning with multiple outputs",
    id: "5c",
};

const CHAPTER5D: Activity = Activity {
    task: chapter5d,
    name: "Gradient descent learning with multiple inputs and outputs",
    id: "5d",
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
        *el += j[idx];
    }
}

fn ele_mul(s: f64, v: &Vec<f64>) -> Vec<f64> {
    v.iter().map(|e| e * s).collect()
}

fn chapter5a() -> Result<(), std::io::Error> {
    struct Output {
        weights: Vec<f64>,
        prediction: f64,
        iterations: i16,
        error: f64,
    }

    const ALPHA: f64 = 0.01;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i16 = 2000;

    fn solve_weights(input: &Vec<f64>, weights: &Vec<f64>, true_value: f64) -> Output {
        let mut weights = weights.to_owned();

        let mut error = MAX;
        let mut pred = MAX;

        for iteration in 0..MAX_ITERATIONS {
            pred = dot(input, &weights);
            let delta = pred - true_value;
            error = delta * delta;

            if error < TOLERANCE {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error,
                };
            }

            let weight_deltas = ele_mul(-1.0 * delta * ALPHA, input);
            add(&mut weights, &weight_deltas);
        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error,
        };
    }
    let win_or_lose_binary: Vec<f64> = vec![1.0, 1.0, 0.0, 1.0];
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
        error: f64,
    }

    const ALPHA: f64 = 0.3;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i16 = 2000;

    fn solve_weights(input: &Vec<f64>, weights: &Vec<f64>, true_value: f64) -> Output {
        let mut weights = weights.to_owned();

        let mut error = MAX;
        let mut pred = MAX;

        for iteration in 0..MAX_ITERATIONS {
            pred = dot(input, &weights);
            let delta = pred - true_value;
            error = delta * delta;

            if error < TOLERANCE {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error,
                };
            }

            let mut weight_deltas = ele_mul(-1.0 * delta * ALPHA, input);
            weight_deltas[0] = 0.0;
            add(&mut weights, &weight_deltas);
        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error,
        };
    }
    let win_or_lose_binary: Vec<f64> = vec![1.0, 1.0, 0.0, 1.0];
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

fn chapter5c() -> Result<(), std::io::Error> {
    struct Output {
        weights: Vec<f64>,
        prediction: Vec<f64>,
        iterations: i16,
        error: Vec<f64>,
    }

    const ALPHA: f64 = 0.1;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i16 = 2000;

    fn solve_weights(input: f64, weights: &Vec<f64>, true_values: &Vec<f64>) -> Output {
        let mut weights = weights.to_owned();

        let mut error: Vec<f64> = vec![];
        let mut pred: Vec<f64> = vec![];

        for iteration in 0..MAX_ITERATIONS {
            pred = ele_mul(input, &weights);
            let delta: Vec<f64> = pred
                .iter()
                .zip(true_values.iter())
                .map(|(p, t)| p - t)
                .collect();
            error = delta.iter().map(|d| d * d).collect();

            if error.iter().all(|e| *e < TOLERANCE) {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error,
                };
            }

            let weight_deltas = ele_mul(-1.0 * input * ALPHA, &delta);
            add(&mut weights, &weight_deltas);
        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error,
        };
    }
    let hurt: Vec<f64> = vec![0.1, 0.0, 0.0, 0.9];
    let win: Vec<f64> = vec![1.0, 1.0, 0.0, 1.0];
    let sad: Vec<f64> = vec![0.1, 0.0, 0.1, 0.2];

    let wlrec = vec![0.65, 0.8, 0.8, 0.9];

    let weights = vec![0.3, 0.2, 0.9];

    for (wl, h, w, s) in izip!(wlrec, hurt, win, sad) {
        let input = wl;
        let true_values = vec![h, w, s];
        let res = solve_weights(input, &weights, &true_values);
        println!("Input: {:?}, True values: {:?}, Prediction: {:?}, Error: {:?}, Weights: {:?}, Iterations: {}", input, true_values, res.prediction, res.error, res.weights, res.iterations);
    }
    Ok(())
}

macro_rules!vec2d {
    [ $( [ $( $d:expr ),* ] ),* ] => {
        vec![
            $(
                vec![$($d),*],
            )*
        ]
    }
}

fn mult_vect_matrix(vect: &Vec<f64>, mat: &Vec<Vec<f64>>) -> Vec<f64> {
    mat.iter().map(|v| dot(v, vect)).collect()
}

fn outer_prod(a: &Vec<f64>, b: &Vec<f64>) -> Vec<Vec<f64>> {
    let mut res: Vec<Vec<f64>> = vec![];
    for i in 0..(*a).len() {
        let mut row: Vec<f64> = vec![];
        for j in 0..(*b).len() {
            row.push(a[i] * b[j])
        }
        res.push(row)
    }
    res
}

fn mat_add(a: &mut Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> () {
    if a.len() != b.len() {
        panic!("input matrix must be of same size")
    }
    for i in 0..a.len() {
        if a[i].len() != b[i].len() {
            panic!("input matrix must be of same size")
        }
        for j in 0..a[i].len() {
            a[i][j] += b[i][j]
        }
    }
}

fn ele_mul_mat(s: f64, m: &mut Vec<Vec<f64>>) -> () {
    for i in 0..m.len() {
        for j in 0..m[i].len() {
            m[i][j] = m[i][j] * s;
        }
    }
}

fn chapter5d() -> Result<(), std::io::Error> {
    struct Output {
        weights: Vec<Vec<f64>>,
        prediction: Vec<f64>,
        iterations: i32,
        error: Vec<f64>,
    }

    const ALPHA: f64 = 0.01;
    const TOLERANCE: f64 = 0.000000001;
    const MAX_ITERATIONS: i32 = 200000;

    fn solve_weights(input: &Vec<f64>, weights: &Vec<Vec<f64>>, true_values: &Vec<f64>) -> Output {
        let mut weights = weights.to_owned();

        let mut error: Vec<f64> = vec![];
        let mut pred: Vec<f64> = vec![];

        for iteration in 0..MAX_ITERATIONS {
            pred = mult_vect_matrix(input, &weights);
            let delta: Vec<f64> = pred
                .iter()
                .zip(true_values.iter())
                .map(|(p, t)| p - t)
                .collect();
            error = delta.iter().map(|d| d * d).collect();

            if error.iter().all(|e| *e < TOLERANCE) {
                return Output {
                    weights,
                    prediction: pred,
                    iterations: iteration,
                    error,
                };
            }

            let mut weight_deltas = outer_prod(input, &delta);
            ele_mul_mat(-1.0 * ALPHA, &mut weight_deltas);
            // let weight_deltas = ele_mul(-1.0*input*ALPHA, &delta);
            mat_add(&mut weights, &weight_deltas);
        }

        return Output {
            weights,
            prediction: pred,
            iterations: MAX_ITERATIONS,
            error,
        };
    }
    let hurt: Vec<f64> = vec![0.1, 0.0, 0.0, 0.9];
    let win: Vec<f64> = vec![1.0, 1.0, 0.0, 1.0];
    let sad: Vec<f64> = vec![0.1, 0.0, 0.1, 0.2];

    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let weights = vec2d![[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]];

    for (to, wl, nf, h, w, s) in izip!(toes, wlrec, nfans, hurt, win, sad) {
        let input = vec![to, wl, nf];
        let true_values = vec![h, w, s];
        let res = solve_weights(&input, &weights, &true_values);
        println!("Input: {:?}, True values: {:?}, Prediction: {:?}, Error: {:?}, Weights: {:?}, Iterations: {}", input, true_values, res.prediction, res.error, res.weights, res.iterations);
    }
    Ok(())
}
