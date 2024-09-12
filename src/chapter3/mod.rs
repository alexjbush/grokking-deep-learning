use crate::Activity;
use itertools::{izip, Itertools};

macro_rules!vec2d {
    [ $( [ $( $d:expr ),* ] ),* ] => {
        vec![
            $(
                vec![$($d),*],
            )*
        ]
    }
}

pub const CHAPTER3A: Activity = Activity {
    task: chapter3a,
    name: "Simple prediction",
    id: "3a",
};

fn chapter3a() -> Result<(), std::io::Error> {
    fn neural_network(input: f64, weight: f64) -> f64 {
        input * weight
    }

    let weight = 0.1;
    let number_of_toes = vec![8.5, 9.5, 10.0, 9.0];
    for toes in number_of_toes {
        let pred = neural_network(toes, weight);
        println!("Input: {}, Prediction: {:.2}", toes, pred);
    }
    Ok(())
}

pub const CHAPTER3B: Activity = Activity {
    task: chapter3b,
    name: "Making a prediction with multiple inputs",
    id: "3b",
};

fn dot(i: &Vec<f64>, j: &Vec<f64>) -> f64 {
    i.into_iter()
        .zip(j.into_iter())
        .fold(0.0, |acc, (a, b)| acc + a * b)
}

fn chapter3b() -> Result<(), std::io::Error> {
    fn neural_network(input: &Vec<f64>, weights: &Vec<f64>) -> f64 {
        if input.len() != weights.len() {
            panic!("input and weights must be of same length")
        }
        dot(input, weights)
    }
    let weights = vec![0.1, 0.2, 0.0];
    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    for (t, w, n) in izip!(toes, wlrec, nfans) {
        let input = vec![t, w, n];
        let pred = neural_network(&input, &weights);
        println!("Input: {:?}, Prediction: {:.2}", input, pred);
    }
    Ok(())
}

fn mult_vect_matrix(vect: &Vec<f64>, mat: &Vec<Vec<f64>>) -> Vec<f64> {
    mat.iter().map(|v| dot(v, vect)).collect()
}

pub const CHAPTER3C: Activity = Activity {
    task: chapter3c,
    name: "Making a prediction with multiple inputs and outputs",
    id: "3c",
};

fn chapter3c() -> Result<(), std::io::Error> {
    fn neural_network(input: &Vec<f64>, weights: &Vec<Vec<f64>>) -> Vec<f64> {
        for w in weights {
            if input.len() != w.len() {
                panic!("input and weights must be of same length")
            }
        }
        mult_vect_matrix(input, weights)
    }

    let weights: Vec<Vec<f64>> = vec2d![[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]];

    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    for (t, w, n) in izip!(toes, wlrec, nfans) {
        let input = vec![t, w, n];
        let pred = neural_network(&input, &weights);
        println!(
            "Input: {:?}, Prediction(s): {:?}",
            input,
            pred.iter().map(|i| format!("{:.2}", i)).collect_vec(),
        );
    }
    Ok(())
}

pub const CHAPTER3D: Activity = Activity {
    task: chapter3d,
    name: "Making a prediction with multiple layers",
    id: "3d",
};

fn chapter3d() -> Result<(), std::io::Error> {
    fn neural_network(input: &Vec<f64>, weights: &Vec<Vec<Vec<f64>>>) -> Vec<Vec<f64>> {
        let mut i = input.to_owned();
        let mut res: Vec<Vec<f64>> = vec![];
        for layer in weights {
            for w in layer {
                if input.len() != w.len() {
                    panic!("input and weights must be of same length")
                }
            }
            let r = mult_vect_matrix(&i, layer);
            i = r.clone();
            res.push(r);
        }
        res
    }

    let ih_wgt: Vec<Vec<f64>> = vec2d![[0.1, 0.2, -0.1], [-0.1, 0.1, 0.9], [0.1, 0.4, 0.1]];
    let hp_wgt: Vec<Vec<f64>> = vec2d![[0.3, 1.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]];
    let weights: Vec<Vec<Vec<f64>>> = vec![ih_wgt, hp_wgt];

    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    for (t, w, n) in izip!(toes, wlrec, nfans) {
        let input = vec![t, w, n];
        let pred = neural_network(&input, &weights);
        println!("Input: {:?}", input);
        for l in pred {
            println!(
                "Prediction(s): {:?}",
                l.iter().map(|i| format!("{:.3}", i)).collect_vec(),
            );
        }
    }
    Ok(())
}


pub const CHAPTER3E: Activity = Activity {
    task: chapter3e,
    name: "Making a prediction with multiple layers using ndarray",
    id: "3e",
};

fn chapter3e() -> Result<(), std::io::Error> {
    todo!()
}