use std::path::Path;
use std::time::{Duration, Instant};
use crate::utils::{download_files, BASE_PATH, GROKKING_BASE_URL, read_lines};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use crate::{Activity, Chapter};
use ndarray::{
    array, concatenate, s, Array, Array2, Array5, ArrayBase, Axis, CowRepr, Dim, Dimension,
    OwnedRepr, ViewRepr,
};
const ACTIVITIES: [Activity; 1] = [CHAPTER11A];
use mnist::*;
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Distribution, Uniform},
    RandomExt,
};
use ndarray_stats::QuantileExt;
use rand_chacha::ChaCha8Rng;

pub const FILE_LABELS: &str = "labels.txt";
pub const FILE_LABELS_SIZE: usize= 225000;
pub const FILE_REVIEWS: &str = "reviews.txt";
pub const FILE_REVIEWS_SIZE: usize = 33678267;

const FILES_TO_DOWNLOAD: &[(&str, usize)] = &[
    (FILE_LABELS, FILE_LABELS_SIZE),
    (FILE_REVIEWS, FILE_REVIEWS_SIZE),
];


pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Networks that understand language",
    id: "11",
};

const CHAPTER11A: Activity = Activity {
    task: chapter11a,
    name: "A simple implementation in NDARRAY",
    id: "11a",
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

fn sigmoid(
    m: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
    m.map(|v| return 1.0 / (1.0 + (-1.0 * v).exp()))
}

fn chapter11a() -> Result<(), std::io::Error> {

    download_files(GROKKING_BASE_URL, BASE_PATH, FILES_TO_DOWNLOAD).unwrap();

    let raw_reviews:Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_REVIEWS)).unwrap().flatten().collect();
    let raw_labels :Vec<String>= read_lines(Path::new(BASE_PATH).join(FILE_LABELS)).unwrap().flatten().collect();

    let tokens: Vec<HashSet<&str>> = raw_reviews.iter().map( |s| s.split_whitespace().collect::<HashSet<&str>>()).collect();
    
    let vocab: Vec<&str> = tokens.iter().flat_map(|sent| sent.iter().filter(|word| word.len() > 0)).collect::<HashSet<&&str>>().iter().map(|w| **w).collect();
    
    let word2index: HashMap<&str, usize> = vocab.iter().enumerate().map(|f| (*f.1, f.0)).collect();

    let mut input_dataset: Vec<Vec<usize>> = vec![];
    for sent in tokens {
        let mut sent_indices: Vec<usize> = vec![];
        for word in sent {
            let idx = word2index.get(word);
            if let Some(i) = idx {
                sent_indices.push(*i);
            }
        }
        input_dataset.push(sent_indices.into_iter().collect::<HashSet<usize>>().into_iter().collect())
    }

    let target_dataset: Vec<f64> = raw_labels.into_iter().map(|label| {
        if label == "positive" {
            return 1.0
        } else {
            return 0.0
        }
    }).collect();

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let alpha = 0.01;
    let max_iterations = 2;
    let hidden_size = 100;


    let mut weights_0_1 = Array::random_using(
        (vocab.len(), hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 =
        Array::random_using((hidden_size, 1), Uniform::new(-0.1, 0.1), &mut rng);

    let mut correct = 0;
    let mut total = 0;

    for iteration in 0..max_iterations {

        for i in 0..(input_dataset.len() - 1000) {

            let x = &input_dataset[i];
            let y = target_dataset[i];

            let layer_1 = sigmoid(&weights_0_1.row(i).sum_axis(Axis(0)).insert_axis(Axis(0)));
            let layer_2 = sigmoid(&layer_1.dot(&weights_1_2));

        }

    }

    todo!()
    
}