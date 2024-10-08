use crate::{Activity, Chapter};
use faer::{mat, Mat};
use itertools::{izip, Itertools};
use nalgebra::{dmatrix, DMatrix};

const ACTIVITIES: [Activity; 6] = [CHAPTER3A, CHAPTER3B, CHAPTER3C, CHAPTER3D, CHAPTER3EI, CHAPTER3EII];

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Introduction to neural prediction",
    id: "3",
};

macro_rules!vec2d {
    [ $( [ $( $d:expr ),* ] ),* ] => {
        vec![
            $(
                vec![$($d),*],
            )*
        ]
    }
}

const CHAPTER3A: Activity = Activity {
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

const CHAPTER3B: Activity = Activity {
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

const CHAPTER3C: Activity = Activity {
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

const CHAPTER3D: Activity = Activity {
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

const CHAPTER3EI: Activity = Activity {
    task: chapter3ei,
    name: "Making a prediction with multiple layers using nalgebra",
    id: "3ei",
};

fn chapter3ei() -> Result<(), std::io::Error> {
    fn neural_network(input: &DMatrix<f64>, weights: &Vec<DMatrix<f64>>) -> Vec<DMatrix<f64>> {
        let mut i: DMatrix<f64> = input.to_owned();
        let mut res: Vec<DMatrix<f64>> = vec![];
        for layer in weights {
            let r: DMatrix<f64> = i * layer.transpose();
            i = r.clone();
            res.push(r);
        }
        res
    }

    let ih_wgt: DMatrix<f64> = dmatrix![0.1, 0.2, -0.1; -0.1, 0.1, 0.9; 0.1, 0.4, 0.1];
    let hp_wgt: DMatrix<f64> = dmatrix![0.3, 1.1, -0.3; 0.1, 0.2, 0.0; 0.0, 1.3, 0.1];
    let weights: Vec<DMatrix<f64>> = vec![ih_wgt, hp_wgt];

    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    for (t, w, n) in izip!(toes, wlrec, nfans) {
        let input: DMatrix<f64> = dmatrix![t, w, n];
        let pred = neural_network(&input, &weights);
        println!("Input: {:?}", input.iter().collect_vec());
        for l in pred {
            println!(
                "Prediction(s): {:?}",
                l.iter().map(|i| format!("{:.3}", i)).collect_vec(),
            );
        }
    }
    Ok(())
}


const CHAPTER3EII: Activity = Activity {
    task: chapter3eii,
    name: "Making a prediction with multiple layers using faer",
    id: "3eii",
};

fn chapter3eii() -> Result<(), std::io::Error> {
    fn neural_network(input: &Mat<f64>, weights: &Vec<Mat<f64>>) -> Vec<Mat<f64>> {
        let mut i: Mat<f64> = input.to_owned();
        let mut res: Vec<Mat<f64>> = vec![];
        for layer in weights {
            let r: Mat<f64> = i * layer.transpose();
            i = r.clone();
            res.push(r);
        }
        res
    }

    let ih_wgt: Mat<f64> = mat![[0.1, 0.2, -0.1], [-0.1, 0.1, 0.9], [0.1, 0.4, 0.1]];
    let hp_wgt: Mat<f64> = mat![[0.3, 1.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]];
    let weights: Vec<Mat<f64>> = vec![ih_wgt, hp_wgt];

    let toes = vec![8.5, 9.5, 10.0, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 10.5, 1.0];

    for (t, w, n) in izip!(toes, wlrec, nfans) {
        let input: Mat<f64> = mat![[t, w, n]];
        let pred = neural_network(&input, &weights);
        println!("Input: {:?}", input.row_iter().flat_map(|f|f.iter()).collect_vec());
        for l in pred {
            println!(
                "Prediction(s): {:?}",
                l.row_iter().flat_map(|f|f.iter()).map(|i| format!("{:.3}", i)).collect_vec(),
            );
        }
    }
    Ok(())
}
