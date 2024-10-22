use std::time::{Duration, Instant};
use crate::utils::{download_files, BASE_PATH, GROKKING_BASE_URL};

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

fn chapter11a() -> Result<(), std::io::Error> {

    download_files(GROKKING_BASE_URL, BASE_PATH, FILES_TO_DOWNLOAD).unwrap();
    todo!()
    
}