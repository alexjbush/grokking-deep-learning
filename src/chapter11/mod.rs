use crate::utils::{download_files, read_lines, BASE_PATH, GROKKING_BASE_URL};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::{Activity, Chapter};
use ndarray::{Array, ArrayBase, Axis, Data, Dim, Dimension, Ix1, Ix2, NdFloat, OwnedRepr};
const ACTIVITIES: [Activity; 3] = [CHAPTER11A, CHAPTER11B, CHAPTER11C];
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_chacha::ChaCha8Rng;

pub const FILE_LABELS: &str = "labels.txt";
pub const FILE_LABELS_SIZE: usize = 225000;
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

const CHAPTER11B: Activity = Activity {
    task: chapter11b,
    name: "Comparing word embeddings",
    id: "11b",
};

const CHAPTER11C: Activity = Activity {
    task: chapter11c,
    name: "Filling in the blank",
    id: "11c",
};

fn sigmoid<D> (
    m: &ArrayBase<OwnedRepr<f64>, D>,
) -> ArrayBase<OwnedRepr<f64>, D> 
where D: Dimension
{
    m.map(|v| return 1.0 / (1.0 + (-1.0 * v).exp()))
}

fn outer<A, S1, S2>(a: &ArrayBase<S1, Ix1>, b: &ArrayBase<S2, Ix1>) -> Array<A, Ix2>
where
    A: NdFloat,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
{
    let m = a.len();
    let n = b.len();
    let mut ab = Array::zeros((m, n));

    for row in 0..m {
        for col in 0..n {
            let v = ab.get_mut((row, col)).unwrap();
            *v = *a.get(row).unwrap() * *b.get(col).unwrap();
        }
    }
    return ab;
}

fn chapter11a() -> Result<(), std::io::Error> {
    download_files(GROKKING_BASE_URL, BASE_PATH, FILES_TO_DOWNLOAD).unwrap();

    let raw_reviews: Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_REVIEWS))
        .unwrap()
        .flatten()
        .collect();
    let raw_labels: Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_LABELS))
        .unwrap()
        .flatten()
        .collect();

    let tokens: Vec<HashSet<&str>> = raw_reviews
        .iter()
        .map(|s| s.split_whitespace().collect::<HashSet<&str>>())
        .collect();

    let vocab: Vec<&str> = tokens
        .iter()
        .flat_map(|sent| sent.iter().filter(|word| word.len() > 0))
        .collect::<HashSet<&&str>>()
        .iter()
        .map(|w| **w)
        .collect();

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
        input_dataset.push(
            sent_indices
                .into_iter()
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect(),
        )
    }

    let target_dataset: Vec<f64> = raw_labels
        .into_iter()
        .map(|label| {
            if label == "positive" {
                return 1.0;
            } else {
                return 0.0;
            }
        })
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let alpha = 0.01;
    let max_iterations = 2;
    let hidden_size = 100;

    let mut weights_0_1 = Array::random_using(
        (vocab.len(), hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 = Array::random_using((hidden_size, 1), Uniform::new(-0.1, 0.1), &mut rng);

    let mut correct = 0;
    let mut total = 0;

    for iteration in 0..max_iterations {
        for i in 0..(input_dataset.len() - 1000) {
            let x = &input_dataset[i];
            let y = target_dataset[i];

            let layer_1 = sigmoid(
                &weights_0_1
                    .select(Axis(0), x)
                    .sum_axis(Axis(0))
                    // .insert_axis(Axis(0)),
            );
            let layer_2 = sigmoid(&layer_1.dot(&weights_1_2));

            let layer_2_delta = layer_2 - y;
            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t());

            for j in x {
                let mut row = weights_0_1.row_mut(*j);
                // assert_eq!(layer_1_delta.dim().0, 1);
                row -= &(layer_1_delta.to_owned() * alpha);
            }

            // assert_eq!(layer_1.dim().0, 1);
            // assert_eq!(layer_2_delta.dim().0, 1);
            weights_1_2 = weights_1_2 - (outer(&layer_1, &layer_2_delta) * alpha);

            // assert_eq!(layer_2_delta.dim().1, 1);
            if *layer_2_delta.abs().get((0)).unwrap() < 0.5 {
                correct += 1;
            }
            total += 1;

            if i % 10 == 9 {
                let progress =
                    (i.to_f64().unwrap() / (input_dataset.len() - 1000).to_f64().unwrap()) * 100.0;
                println!(
                    "Iter: {}, Progress: {:.2}%, Training Accuracy: {:.3}, Correct: {}, Total: {}",
                    iteration,
                    progress,
                    correct.to_f64().unwrap() / total.to_f64().unwrap(),
                    correct,
                    total
                );
            }
        }
    }

    correct = 0;
    total = 0;

    for i in (input_dataset.len() - 1000)..input_dataset.len() {
        let x = &input_dataset[i];
        let y = target_dataset[i];

        let layer_1 = sigmoid(
            &weights_0_1
                .select(Axis(0), x)
                .sum_axis(Axis(0))
                // .insert_axis(Axis(0)),
        );
        let layer_2 = sigmoid(&layer_1.dot(&weights_1_2));

        let layer_2_delta = layer_2 - y;

        assert_eq!(layer_2_delta.dim(), 1);
        // assert_eq!(layer_2_delta.dim().1, 1);
        if *(layer_2_delta.abs().get(0).unwrap()) < 0.5 {
            correct += 1;
        }
        total += 1;
    }

    println!(
        "Test Accuracy: {:.3}",
        correct.to_f64().unwrap() / total.to_f64().unwrap()
    );

    Ok(())
}

fn chapter11b() -> Result<(), std::io::Error> {
    download_files(GROKKING_BASE_URL, BASE_PATH, FILES_TO_DOWNLOAD).unwrap();

    let raw_reviews: Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_REVIEWS))
        .unwrap()
        .flatten()
        .collect();
    let raw_labels: Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_LABELS))
        .unwrap()
        .flatten()
        .collect();

    let tokens: Vec<HashSet<&str>> = raw_reviews
        .iter()
        .map(|s| s.split_whitespace().collect::<HashSet<&str>>())
        .collect();

    let vocab: Vec<&str> = tokens
        .iter()
        .flat_map(|sent| sent.iter().filter(|word| word.len() > 0))
        .collect::<HashSet<&&str>>()
        .iter()
        .map(|w| **w)
        .collect();

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
        input_dataset.push(
            sent_indices
                .into_iter()
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect(),
        )
    }

    let target_dataset: Vec<f64> = raw_labels
        .into_iter()
        .map(|label| {
            if label == "positive" {
                return 1.0;
            } else {
                return 0.0;
            }
        })
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let alpha = 0.01;
    let max_iterations = 2;
    let hidden_size = 100;

    let mut weights_0_1 = Array::random_using(
        (vocab.len(), hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 = Array::random_using((hidden_size, 1), Uniform::new(-0.1, 0.1), &mut rng);

    for _ in 0..max_iterations {
        for i in 0..(input_dataset.len() - 1000) {
            let x = &input_dataset[i];
            let y = target_dataset[i];

            let layer_1 = sigmoid(
                &weights_0_1
                    .select(Axis(0), x)
                    .sum_axis(Axis(0))
            );
            let layer_2 = sigmoid(&layer_1.dot(&weights_1_2));

            let layer_2_delta = layer_2 - y;
            let layer_1_delta = layer_2_delta.dot(&weights_1_2.t());

            for j in x {
                let mut row = weights_0_1.row_mut(*j);
                assert_eq!(layer_1_delta.dim(), 1);
                row -= &(layer_1_delta.to_owned() * alpha);
            }

            assert_eq!(layer_1.dim(), 1);
            assert_eq!(layer_2_delta.dim(), 1);
            weights_1_2 = weights_1_2 - (outer(&layer_1, &layer_2_delta) * alpha);

        }
    }

    fn similar<'a>(
        target: &str,
        word2index: &HashMap<&'a str, usize>,
        weights_0_1: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ) -> Vec<(&'a str, f64)> {
        let target_index = word2index[target];
        let mut scores: Vec<(&str, f64)> = vec![];
        for (word, index) in word2index.iter() {
            let raw_difference = &weights_0_1.row(*index) - &weights_0_1.row(target_index);
            let squared_difference = &raw_difference * &raw_difference;
            scores.push((*word, -1.0 * (squared_difference.sum().sqrt())));
        }
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        return scores.into_iter().take(10).collect();
    }

    println!("{:#?}", similar("beautiful", &word2index, &weights_0_1));

    println!("{:#?}", similar("terrible", &word2index, &weights_0_1));

    Ok(())
}

fn chapter11c() -> Result<(), std::io::Error> {
    download_files(GROKKING_BASE_URL, BASE_PATH, FILES_TO_DOWNLOAD).unwrap();

    let raw_reviews: Vec<String> = read_lines(Path::new(BASE_PATH).join(FILE_REVIEWS))
        .unwrap()
        .flatten()
        .collect();

    let tokens: Vec<Vec<&str>> = raw_reviews
        .iter()
        .map(|s| s.split(' ').collect::<Vec<&str>>())
        .collect();

    let mut wordcnt: HashMap<&str, isize> = HashMap::new();
    for sent in &tokens {
        for word in sent {
            if let Some(x) = wordcnt.get_mut(word) {
                *x -= 1;
            } else {
                wordcnt.insert(word, -1);
            }
        }
    }
    let mut _wordcnt: Vec<(&str, isize)> = wordcnt.into_iter().collect();
    _wordcnt.sort_by(|(_, a), (_, b)| b.cmp(a));
    let wordcnt: Vec<(&str, isize)> = _wordcnt.into_iter().collect();

    let vocab: Vec<&str> = wordcnt
        .into_iter()
        .map(|(word, _)| word)
        .collect::<HashSet<&str>>()
        .into_iter()
        .collect();

    let word2index: HashMap<&str, usize> = (&vocab)
        .into_iter()
        .enumerate()
        .map(|f| (*f.1, f.0))
        .collect();

    let mut concatenated: Vec<usize> = vec![];
    let mut input_dataset: Vec<Vec<usize>> = vec![];
    for sent in &tokens {
        let mut sent_indices: Vec<usize> = vec![];
        for word in sent {
            let idx = word2index.get(word);
            if let Some(i) = idx {
                sent_indices.push(*i);
                concatenated.push(*i);
            }
        }
        input_dataset.push(
            sent_indices
                .into_iter()
                .collect::<HashSet<usize>>()
                .into_iter()
                .collect(),
        )
    }

    let concatenated = Array::from_vec(concatenated);

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    input_dataset.shuffle(&mut rng);

    let alpha = 0.05;
    let max_iterations = 2;
    let hidden_size = 50;
    let window = 2;
    let negative = 5;

    let mut weights_0_1 = Array::random_using(
        ((&vocab).len(), hidden_size),
        Uniform::new(-0.1, 0.1),
        &mut rng,
    );
    let mut weights_1_2 = Array::zeros(((&vocab).len(), hidden_size));

    let mut layer_2_target = Array::zeros(negative + 1);
    *layer_2_target.get_mut(0).unwrap() = 1.0;

    fn similar<'a>(
        target: &str,
        word2index: &HashMap<&'a str, usize>,
        weights_0_1: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    ) -> Vec<(&'a str, f64)> {
        let target_index = word2index[target];
        let mut scores: Vec<(&str, f64)> = vec![];
        for (word, index) in word2index.iter() {
            let raw_difference = &weights_0_1.row(*index) - &weights_0_1.row(target_index);
            let squared_difference = &raw_difference * &raw_difference;
            scores.push((*word, -1.0 * (squared_difference.sum().sqrt())));
        }
        scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        return scores.into_iter().take(10).collect();
    }

    for rev_i in 0..(input_dataset.len() * max_iterations) {
        let review = &input_dataset[rev_i % (input_dataset.len())];
        for target_i in 0..((*review).len()) {
            
            let indexes =
                Array::random_using(negative, Uniform::new(0, concatenated.len()), &mut rng)
                    .to_vec();
            let target_samples = [
                vec![review[target_i]],
                concatenated.select(Axis(0), &indexes).to_vec(),
            ]
            .concat();

            let left_context = &review[0.max(if window < target_i {
                target_i - window
            } else {
                0
            })..target_i];
            let right_context = &review[target_i + 1..review.len().min(target_i + window)];

            let full_context = [left_context, right_context].concat();

            let layer_1 = weights_0_1
                .select(Axis(0), &full_context)
                .mean_axis(Axis(0))
                .unwrap();

            let layer_2 = sigmoid(&layer_1.dot(&weights_1_2.select(Axis(0), &target_samples).t()));

            let layer_2_delta = layer_2 - &layer_2_target;
            let layer_1_delta = layer_2_delta.dot(&weights_1_2.select(Axis(0), &target_samples));

            for i in 0..full_context.len() {
                let j = full_context[i];
                let mut row = weights_0_1.row_mut(j);
                assert_eq!(layer_1_delta.dim(), 1);
                row -= &((layer_1_delta).to_owned() * alpha);
            }

            assert_eq!(layer_1.dim(), 1);
            assert_eq!(layer_2_delta.dim(), 1);
            let weights_1_2_delta = outer(&layer_2_delta, &layer_1) * alpha;
            for i in 0..target_samples.len() {
                let j = target_samples[i];
                let mut row = weights_1_2.row_mut(j);
                row -= &weights_1_2_delta.row(i);
            }
        }

        if rev_i % 1000 == 0 {
            let progress = (rev_i.to_f64().unwrap()
                / (input_dataset.len() * max_iterations).to_f64().unwrap())
                * 100.0;
            println!("Progress: {:.2}%, i: {}", progress, rev_i);
            println!("{:#?}", similar("terrible", &word2index, &weights_0_1));
        }
    }

    println!("{:#?}", similar("terrible", &word2index, &weights_0_1));
    println!("{:#?}", similar("beautiful", &word2index, &weights_0_1));

    Ok(())
}
