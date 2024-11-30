use crate::{Activity, Chapter};
use ndarray::array;
use ndarray::Array;
use ndarray::ArrayD;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::Ix1;
use ndarray::Ix2;
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_chacha::ChaCha8Rng;
use std::iter;
use std::{
    collections::{HashMap, HashSet},
    ops::{self},
};

use uuid::Uuid;
pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Introducing automatic optimisation",
    id: "13",
};

const CHAPTER13A: Activity = Activity {
    task: chapter13a,
    name: "Introduction to Autograd",
    id: "13a",
};

const CHAPTER13B: Activity = Activity {
    task: chapter13b,
    name: "Adding support for negation",
    id: "13b",
};

const CHAPTER13C: Activity = Activity {
    task: chapter13c,
    name: "Using autograd to train a neural network",
    id: "13c",
};

const ACTIVITIES: [Activity; 3] = [CHAPTER13A, CHAPTER13B, CHAPTER13C];

trait Operation {
    fn deriv(&self, grad: &Tensor<f64>, creators: &Vec<&Tensor<f64>>)
        -> HashMap<Uuid, Tensor<f64>>;
}

struct Add();
struct Neg();
struct Sub();
struct Mul();
struct Sum {
    dim: usize,
}
struct Expand {
    dim: usize,
}
struct Transpose();
struct MM();

impl Operation for Add {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 2);
        creators
            .iter()
            .map(|t| (t.id, Tensor::new(grad.data.clone(), false)))
            .collect()
    }
}

impl Operation for Neg {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        HashMap::from([(creators[0].id, Tensor::new(-&grad.data, false))])
    }
}

impl Operation for Sub {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 2);
        HashMap::from([
            (creators[0].id, Tensor::new(grad.data.clone(), false)),
            (creators[1].id, Tensor::new(-&grad.data, false)),
        ])
    }
}

impl Operation for Mul {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 2);
        HashMap::from([
            (
                creators[0].id,
                Tensor::new(&grad.data * &creators[1].data, false),
            ),
            (
                creators[1].id,
                Tensor::new(&grad.data * &creators[0].data, false),
            ),
        ])
    }
}

impl Operation for MM {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 2);
        let act = creators[0];
        let weights = creators[1];
        let w_t = weights.transpose();
        let new = grad.mm(&w_t);
        let g_t = grad.transpose();
        let g_t__mm__act = g_t.mm(act);
        let new_new = g_t__mm__act.transpose();
        HashMap::from([
            (act.id, Tensor::new(new.data, false)),
            (weights.id, Tensor::new(new_new.data, false)),
        ])
    }
}

impl Operation for Sum {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        let dim = self.dim;
        let ds = creators[0].data.dim()[dim];
        let new = grad.expand(dim, ds);
        HashMap::from([(creators[0].id, Tensor::new(new.data, false))])
    }
}

impl Operation for Expand {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        let dim = self.dim;
        let new = grad.sum(dim);
        HashMap::from([(creators[0].id, Tensor::new(new.data, false))])
    }
}

impl Operation for Transpose {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        HashMap::from([(creators[0].id, Tensor::new(grad.data.t().to_owned(), false))])
    }
}

pub struct Gradients<'a> {
    grads: HashMap<Uuid, Tensor<'a, f64>>,
}

pub struct Tensor<'a, T> {
    id: Uuid,
    data: ArrayD<T>,
    autograd: bool,
    creators: Vec<&'a Tensor<'a, T>>,
    creation_op: Option<Box<dyn Operation>>,
}

impl<'a, T> Tensor<'a, T> {
    pub fn new(data: ArrayD<T>, autograd: bool) -> Tensor<'a, T> {
        Tensor {
            id: Uuid::new_v4(),
            data,
            autograd,
            creators: vec![],
            creation_op: None,
        }
    }

    fn new_with_creators(
        data: ArrayD<T>,
        creators: Vec<&'a Tensor<'a, T>>,
        creation_op: Option<Box<dyn Operation>>,
    ) -> Tensor<'a, T> {
        Tensor {
            id: Uuid::new_v4(),
            data,
            autograd: true,
            creators,
            creation_op,
        }
    }

    pub fn get_data(&self) -> &ArrayD<T> {
        return &self.data;
    }
}

impl<'a> Tensor<'a, f64> {
    pub fn get_gradient<'b>(&self, grads: &'b Gradients<'b>) -> &'b Tensor<'b, f64> {
        return &grads.grads.get(&self.id).unwrap();
    }

    pub fn backwards(&self, grad: &Tensor<'a, f64>) -> Gradients {
        // let mut to_visit: Vec<(&Tensor<T>, HashSet<Uuid>)> = self
        //     .creators
        //     .iter()
        //     .map(|v| (*v, HashSet::from([self.id])))
        //     .collect();

        let mut to_visit: Vec<(&Tensor<f64>, HashSet<Uuid>)> = vec![(self, HashSet::new())];

        let mut depends_on: HashMap<Uuid, Vec<&Tensor<f64>>> = HashMap::from([(self.id, vec![])]);

        let mut grads = Gradients {
            grads: HashMap::from([(self.id, Tensor::new(grad.data.clone(), false))]),
        };

        let mut id_to_tensor: HashMap<Uuid, &Tensor<f64>> = HashMap::new();

        while let Some((t, h)) = to_visit.pop() {
            assert!(t.autograd);
            id_to_tensor.insert(t.id, t);
            t.creators.iter().for_each(|v| {
                if h.contains(&t.id) {
                    panic!(
                        "Graph is cyclic, tensor with id {} found multiple times",
                        t.id
                    );
                }
                let mut new_set = h.clone();
                new_set.insert(t.id);
                to_visit.push((v, new_set));

                if let Some(d) = depends_on.get_mut(&v.id) {
                    d.push(t);
                } else {
                    depends_on.insert(v.id, vec![t]);
                }
            });
        }

        while !depends_on.is_empty() {
            let maybe_t = depends_on
                .iter()
                .find(|(_, contributors)| {
                    contributors.iter().all(|t| !depends_on.contains_key(&t.id))
                })
                .map(|(id, _)| *id);

            let deps = depends_on.keys().collect::<Vec<&Uuid>>();
            if let Some(id) = maybe_t {
                depends_on.remove(&id).unwrap();

                let this = id_to_tensor.get(&id).unwrap();

                if !this.creators.is_empty() {
                    let op: &Box<dyn Operation> = this.creation_op.as_ref().unwrap();

                    let res = op.deriv(this.get_gradient(&grads), &this.creators);
                    for (id, g) in res {
                        if let Some(gg) = grads.grads.get(&id) {
                            grads
                                .grads
                                .insert(id, Tensor::new(&gg.data + g.data, false));
                        } else {
                            grads.grads.insert(id, g);
                        }
                    }
                }
            } else {
                panic!(
                    "Could not resolve all dependencies for remaining tensors: {:?}",
                    deps
                );
            }
        }
        return grads;
    }

    pub fn sum(&'a self, dim: usize) -> Tensor<'a, f64> {
        let data = self.get_data().sum_axis(ndarray::Axis(dim));
        if self.autograd {
            Tensor::new_with_creators(data, vec![self], Some(Box::new(Sum { dim })))
        } else {
            Tensor::new(data, false)
        }
    }

    pub fn transpose(&'a self) -> Tensor<'a, f64> {
        let data = self.get_data().t().to_owned();
        if self.autograd {
            Tensor::new_with_creators(data, vec![self], Some(Box::new(Transpose())))
        } else {
            Tensor::new(data, false)
        }
    }

    pub fn expand(&'a self, dim: usize, copies: usize) -> Tensor<'a, f64> {
        let mut trans_cmd: Vec<usize> = (0..self.data.ndim()).collect();
        trans_cmd.insert(dim, self.data.ndim());
        let new_shape = vec![self.data.dim().as_array_view().to_vec(), vec![copies]].concat();
        // println!("Old: {:?}, New: {:?}", self.data.dim(), new_shape);
        let _new_data: Vec<f64> = self
            .data
            .iter()
            .flat_map(|v| iter::repeat(*v).take(copies))
            .collect();
        let data = ArrayD::from_shape_vec(new_shape, _new_data)
            .unwrap()
            .permuted_axes(trans_cmd);
        if self.autograd {
            Tensor::new_with_creators(data, vec![self], Some(Box::new(Expand { dim })))
        } else {
            Tensor::new(data, false)
        }
    }

    pub fn mm(&'a self, _rhs: &'a Self) -> Tensor<'a, f64> {
        let data = if self.data.ndim() == 1 && _rhs.data.ndim() == 1 {
            array![self
                .data
                .to_owned()
                .into_dimensionality::<Ix1>()
                .unwrap()
                .dot(&_rhs.data.to_owned().into_dimensionality::<Ix1>().unwrap())]
            .into_dyn()
        } else if self.data.ndim() == 1 && _rhs.data.ndim() == 2 {
            self.data
                .to_owned()
                .into_dimensionality::<Ix1>()
                .unwrap()
                .dot(&_rhs.data.to_owned().into_dimensionality::<Ix2>().unwrap())
                .into_dyn()
        } else if self.data.ndim() == 2 && _rhs.data.ndim() == 1 {
            self.data
                .to_owned()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .dot(&_rhs.data.to_owned().into_dimensionality::<Ix1>().unwrap())
                .into_dyn()
        } else if self.data.ndim() == 2 && _rhs.data.ndim() == 2 {
            self.data
                .to_owned()
                .into_dimensionality::<Ix2>()
                .unwrap()
                .dot(&_rhs.data.to_owned().into_dimensionality::<Ix2>().unwrap())
                .into_dyn()
        } else {
            panic!(
                "Unsupported dimension size for mm: [{}] and [{}]",
                self.data.ndim(),
                _rhs.data.ndim()
            )
        };
        if self.autograd {
            Tensor::new_with_creators(data, vec![self, _rhs], Some(Box::new(MM())))
        } else {
            Tensor::new(data, false)
        }
    }
}

impl<'a> ops::Add for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn add(self, _rhs: Self) -> Self::Output {
        let data = self.get_data() + _rhs.get_data();
        if self.autograd {
            Tensor::new_with_creators(data, vec![self, _rhs], Some(Box::new(Add())))
        } else {
            Tensor::new(data, false)
        }
    }
}

impl<'a> ops::Neg for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn neg(self) -> Self::Output {
        let data = -(self.get_data());

        if self.autograd {
            Tensor::new_with_creators(data, vec![self], Some(Box::new(Neg())))
        } else {
            Tensor::new(data, false)
        }
    }
}

impl<'a> ops::Sub for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn sub(self, _rhs: Self) -> Self::Output {
        let data = self.get_data() - _rhs.get_data();
        if self.autograd {
            Tensor::new_with_creators(data, vec![self, _rhs], Some(Box::new(Sub())))
        } else {
            Tensor::new(data, false)
        }
    }
}

impl<'a> ops::Mul for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn mul(self, _rhs: Self) -> Self::Output {
        let data = self.get_data() * _rhs.get_data();
        if self.autograd {
            Tensor::new_with_creators(data, vec![self, _rhs], Some(Box::new(Mul())))
        } else {
            Tensor::new(data, false)
        }
    }
}

fn chapter13a() -> Result<(), std::io::Error> {
    let a = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn(), true);
    let b = Tensor::new(array![2.0, 2.0, 2.0, 2.0, 2.0].into_dyn(), true);
    let c = Tensor::new(array![5.0, 4.0, 3.0, 2.0, 1.0].into_dyn(), true);
    // let d = Tensor::new(array![-1.0, -2.0, -3.0, -4.0, -5.0].into_dyn());

    let d = &a + &b;
    let e = &b + &c;
    let f = &d + &e;

    let grads = f.backwards(&Tensor::new(
        array![1.0, 1.0, 1.0, 1.0, 1.0].into_dyn(),
        false,
    ));

    println!("{:?}", &b.get_gradient(&grads).data);

    Ok(())
}

fn chapter13b() -> Result<(), std::io::Error> {
    let a = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn(), true);
    let b = Tensor::new(array![2.0, 2.0, 2.0, 2.0, 2.0].into_dyn(), true);
    let c = Tensor::new(array![5.0, 4.0, 3.0, 2.0, 1.0].into_dyn(), true);
    // let d = Tensor::new(array![-1.0, -2.0, -3.0, -4.0, -5.0].into_dyn());

    let negb1 = -&b;
    let negb2 = -&b;
    let d = &a + &negb1;
    let e = &negb2 + &c;
    let f = &d + &e;

    let grads = f.backwards(&Tensor::new(
        array![1.0, 1.0, 1.0, 1.0, 1.0].into_dyn(),
        false,
    ));

    println!("{:?}", &b.get_gradient(&grads).data);

    Ok(())
}

fn chapter13c() -> Result<(), std::io::Error> {
    let data = Tensor::new(
        array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn(),
        true,
    );
    let target = Tensor::new(array![[0.0], [1.0], [0.0], [1.0]].into_dyn(), true);

    let mut rng = ChaCha8Rng::seed_from_u64(1);

    let mut w0 = Array::random_using((2, 3), Uniform::new(0.0, 1.0), &mut rng).into_dyn();
    let mut w1 = Array::random_using((3, 1), Uniform::new(0.0, 1.0), &mut rng).into_dyn();

    for _ in 0..10 {
        let t0 = Tensor::new(w0.clone(), true);
        let t1 = Tensor::new(w1.clone(), true);

        let g = (&data).mm(&t0);
        let pred = (g).mm(&t1);

        let diff = &pred - &target;
        let double = &diff * &diff;
        let loss = double.sum(0);

        let grad = Tensor::new(Array::ones(loss.data.dim()), false);
        let grads = loss.backwards(&grad);

        let g0 = t0.get_gradient(&grads);
        let g1 = t1.get_gradient(&grads);

        w0 = &w0 - &g0.data * 0.1;
        w1 = &w1 - &g1.data * 0.1;
        println!("{:?}", loss.data);
    }

    Ok(())
}
