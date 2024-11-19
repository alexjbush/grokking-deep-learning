use crate::utils::{download_files, read_lines, BASE_PATH, GROKKING_BASE_URL};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::{Activity, Chapter};
use ndarray::{
    Array, Array1, Array2, ArrayBase, Axis, Data, Dimension, Ix1, Ix2, NdFloat, OwnedRepr,
};
const ACTIVITIES: [Activity; 2] = [CHAPTER12A, CHAPTER12B];
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use ndarray_stats::QuantileExt;
use rand_chacha::ChaCha8Rng;

pub const FILE_LABELS: &str = "labels.txt";
pub const FILE_LABELS_SIZE: usize = 225000;
pub const FILE_REVIEWS: &str = "reviews.txt";
pub const FILE_REVIEWS_SIZE: usize = 33678267;

pub const BABI_FILE: &str = "qa1_single-supporting-fact_train.txt";
pub const BABI_GROKKING_BASE_URL: &str =
    "https://github.com/iamtrask/Grokking-Deep-Learning/raw/refs/heads/master/tasksv11/en";
pub const BABI_FILE_SIZE: usize = 94346;

const FILES_TO_DOWNLOAD: &[(&str, usize)] = &[
    (FILE_LABELS, FILE_LABELS_SIZE),
    (FILE_REVIEWS, FILE_REVIEWS_SIZE),
];

const BABI_FILE_TO_DOWNLOAD: &[(&str, usize)] = &[(BABI_FILE, BABI_FILE_SIZE)];

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Networks that that write like Shakespeare",
    id: "12",
};

const CHAPTER12A: Activity = Activity {
    task: chapter12a,
    name: "The surprising power of averaged word vectors",
    id: "12a",
};

const CHAPTER12B: Activity = Activity {
    task: chapter12b,
    name: "Learning the transition matrices",
    id: "12b",
};

fn sigmoid<D>(m: &ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>
where
    D: Dimension,
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

fn chapter12a() -> Result<(), std::io::Error> {
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
                .collect::<Vec<usize>>()
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
                row -= &((layer_1_delta).to_owned() * alpha);
            }

            let weights_1_2_delta = outer(&layer_2_delta, &layer_1) * alpha;
            for i in 0..target_samples.len() {
                let j = target_samples[i];
                let mut row = weights_1_2.row_mut(j);
                row -= &weights_1_2_delta.row(i);
            }
        }
    }

    let _norms = (&weights_0_1 * &weights_0_1).sum_axis(Axis(1));
    let norms = _norms.to_shape((_norms.dim(), 1)).unwrap();
    let normed_weights = weights_0_1 * norms;

    fn make_sent_vect(
        words: &Vec<&str>,
        word2index: &HashMap<&str, usize>,
        normed_weights: &Array2<f64>,
    ) -> Array1<f64> {
        let indices: Vec<usize> = words
            .into_iter()
            .flat_map(|v| word2index.get(*v).map(|v| *v))
            .collect();
        return normed_weights
            .select(Axis(0), &indices)
            .mean_axis(Axis(0))
            .unwrap();
    }

    let _reviews2vector: Vec<Array1<f64>> = tokens
        .into_iter()
        .map(|review| make_sent_vect(&review, &word2index, &normed_weights))
        .collect();

    let _reviews2vector = Array::from_vec(_reviews2vector);
    let _inner_shape = _reviews2vector[0].dim();
    let _shape = (_reviews2vector.len(), _inner_shape);
    let _flat: Vec<f64> = _reviews2vector.iter().flatten().cloned().collect();
    let reviews2vector = Array2::from_shape_vec(_shape, _flat).unwrap();

    fn most_similar_reviews<'a>(
        words: &Vec<&str>,
        reviews2vector: &Array2<f64>,
        word2index: &HashMap<&str, usize>,
        normed_weights: &Array2<f64>,
        raw_reviews: &'a Vec<String>,
    ) -> Vec<&'a str> {
        let v = make_sent_vect(words, word2index, normed_weights);

        let mut _scores_most_common: Vec<(usize, f64)> =
            reviews2vector.dot(&v).into_iter().enumerate().collect();
        _scores_most_common.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        println!("{:?}", &_scores_most_common[0..5]);

        return _scores_most_common
            .into_iter()
            .map(|(idx, _)| idx)
            .take(3)
            .map(|idx| &(raw_reviews[idx][0..40]))
            .collect();
    }

    println!(
        "{:#?}",
        most_similar_reviews(
            &vec!["boring", "awful"],
            &reviews2vector,
            &word2index,
            &normed_weights,
            &raw_reviews
        )
    );

    Ok(())
}

fn softmax<D>(m: &ArrayBase<OwnedRepr<f64>, D>) -> ArrayBase<OwnedRepr<f64>, D>
where
    D: Dimension + ndarray::RemoveAxis,
{
    let e_x = (m - *m.max().unwrap()).exp();
    return &e_x / &e_x.sum_axis(Axis(0));
}

fn chapter12b() -> Result<(), std::io::Error> {
    download_files(BABI_GROKKING_BASE_URL, BASE_PATH, BABI_FILE_TO_DOWNLOAD).unwrap();

    let raw: Vec<String> = read_lines(Path::new(BASE_PATH).join(BABI_FILE))
        .unwrap()
        .flatten()
        .collect();

    let _tokens: Vec<Vec<String>> = raw
        .into_iter()
        .take(1000)
        .map(|line| {
            line.to_lowercase()
                .replace('\n', "")
                .split(' ')
                .map(|s| s.to_owned())
                .skip(1)
                .collect::<Vec<String>>()
        })
        .collect();

    let tokens: Vec<Vec<&str>> = _tokens
        .iter()
        .map(|sent| sent.into_iter().map(|word| &word[..]).collect())
        .collect();

    let vocab = tokens
        .iter()
        .flat_map(|sent| sent)
        .map(|v| *v)
        .collect::<HashSet<&str>>()
        .into_iter()
        .collect::<Vec<&str>>();
    let word2index: HashMap<&str, usize> = vocab.iter().enumerate().map(|f| (*f.1, f.0)).collect();

    fn words2indices(sentence: &Vec<&str>, word2index: &HashMap<&str, usize>) -> Vec<usize> {
        return sentence.into_iter().map(|word| word2index[word]).collect();
    }

    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let embed_size = 10;

    let mut embed = Array::random_using(
        ((&vocab).len(), embed_size),
        Uniform::new(-0.05, 0.05),
        &mut rng,
    );

    let mut recurrent: Array2<f64> = Array::eye(embed_size);

    let mut start: Array1<f64> = Array::zeros(embed_size);

    let mut decoder = Array::random_using(
        (embed_size, (&vocab).len()),
        Uniform::new(-0.05, 0.05),
        &mut rng,
    );

    let one_hot: Array2<f64> = Array::eye((&vocab).len());

    fn predict(
        sent: &Vec<usize>,
        start: &Array1<f64>,
        decoder: &Array2<f64>,
        recurrent: &Array2<f64>,
        embed: &Array2<f64>,
    ) -> (Vec<HashMap<&'static str, Array1<f64>>>, f64) {
        let mut layers: Vec<HashMap<&str, Array1<f64>>> =
            vec![HashMap::from([("hidden", start.to_owned())])];

        let mut loss = 0.0;

        for target_i in 0..sent.len() {
            let mut layer: HashMap<&str, Array1<f64>> = HashMap::new();
            layer.insert(
                "pred",
                softmax(&layers.last().unwrap()["hidden"].dot(decoder)),
            );

            loss += -(layer["pred"].get(sent[target_i]).unwrap()).ln();

            layer.insert(
                "hidden",
                &layers.last().unwrap()["hidden"].dot(recurrent) + &embed.row(sent[target_i]),
            );

            layers.push(layer);
        }

        return (layers, loss);
    }

    for iter in 0..30000 {
        let alpha = 0.001;
        let sent = words2indices(&tokens[iter % tokens.len()][1..].to_vec(), &word2index);
        let (mut layers, loss) = predict(&sent, &start, &decoder, &recurrent, &embed);

        for layer_idx in (0..layers.len()).rev() {
            let layers_len = layers.len();
            let layer = layers.get(layer_idx).unwrap();
            let mut new_layer: HashMap<&str, Array1<f64>> = HashMap::new();
            let send_idx = (if layer_idx == 0 {
                sent.len()
            } else {
                layer_idx
            }) - 1;
            // let target = sent[layer_idx - 1];
            let target = sent[send_idx];

            if layer_idx > 0 {
                let output_delta = &layer["pred"] - &one_hot.row(target);
                let new_hidden_delta = output_delta.dot(&decoder.t());

                if layer_idx == layers_len - 1 {
                    new_layer.insert("hidden_delta", new_hidden_delta);
                } else {
                    let _d = &layers
                        .get(layer_idx + 1)
                        .unwrap()
                        .get("hidden_delta")
                        .unwrap()
                        .dot(&recurrent.t());
                    new_layer.insert("hidden_delta", new_hidden_delta + _d);
                }
                new_layer.insert("output_delta", output_delta);
            } else {
                let _d = layers
                    .get(layer_idx + 1)
                    .unwrap()
                    .get("hidden_delta")
                    .unwrap()
                    .dot(&recurrent.t());
                new_layer.insert("hidden_delta", _d);
            }
            let layer_mut = layers.get_mut(layer_idx).unwrap();
            new_layer.iter().for_each(|(k, v)| {
                layer_mut.insert(*k, v.to_owned());
            });
        }

        start -= &(&layers[0]["hidden_delta"] * (alpha / sent.len() as f64));
        for (layer_idx, layer) in (&layers[1..]).iter().enumerate() {
            decoder -= &(outer(&(&layers[layer_idx]["hidden"]), &layer["output_delta"])
                * (alpha / sent.len() as f64));

            let embed_idx = sent[layer_idx];
            let mut row = embed.row_mut(embed_idx);
            row -= &(&layers[layer_idx]["hidden_delta"] * (alpha / sent.len() as f64));

            recurrent -= &(outer(&(&layers[layer_idx]["hidden"]), &layer["hidden_delta"])
                * (alpha / sent.len() as f64));
        }

        if iter % 1000 == 0 {
            println!("Perplexity: {:.3}", (loss / (sent.len() as f64)).exp());
        }
    }

    let send_index = 4;
    let (l, _) = predict(
        &words2indices(&tokens[send_index], &word2index),
        &start,
        &decoder,
        &recurrent,
        &embed,
    );

    println!("{:?}", tokens[send_index]);

    for (i, each_layer) in l[1..l.len() - 1].iter().enumerate() {
        let input = tokens[send_index][i];
        let true_v = tokens[send_index][i + 1];
        let pred = vocab[each_layer["pred"].argmax().unwrap()];
        println!(
            "Prev Input: {:?}{}True: {:?}{}Pred: {:?}",
            input,
            " ".repeat(12 - input.len()),
            true_v,
            " ".repeat(15 - true_v.len()),
            pred,
        );
    }
    Ok(())
}
