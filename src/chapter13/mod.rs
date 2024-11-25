use crate::{Activity, Chapter};
use ndarray::array;
use ndarray::ArrayD;
use num_traits::Float;
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

const ACTIVITIES: [Activity; 1] = [CHAPTER13A];

trait Operation<T: Float> {
    fn deriv(&self, grad: &Tensor<T>) -> Tensor<T>;
}

struct Add();

impl<T: Float> Operation<T> for Add {
    fn deriv(&self, grad: &Tensor<T>) -> Tensor<T> {
        Tensor::new(grad.data.clone())
    }
}

pub struct Gradients<'a, T: Float> {
    grads: HashMap<Uuid, Tensor<'a, T>>,
}

pub struct Tensor<'a, T> {
    id: Uuid,
    data: ArrayD<T>,
    creators: Vec<&'a Tensor<'a, T>>,
    creation_op: Option<Box<dyn Operation<T>>>,
}

impl<'a, T> Tensor<'a, T> {
    pub fn new(data: ArrayD<T>) -> Tensor<'a, T> {
        Tensor {
            id: Uuid::new_v4(),
            data,
            creators: vec![],
            creation_op: None,
        }
    }

    fn new_with_creators(
        data: ArrayD<T>,
        creators: Vec<&'a Tensor<'a, T>>,
        creation_op: Option<Box<dyn Operation<T>>>,
    ) -> Tensor<'a, T> {
        Tensor {
            id: Uuid::new_v4(),
            data,
            creators,
            creation_op,
        }
    }

    pub fn get_data(&self) -> &ArrayD<T> {
        return &self.data;
    }
}

impl<'a, T> Tensor<'a, T>
where
    T: Float + Clone,
{
    pub fn get_gradient<'b>(&self, grads: &'b Gradients<'b, T>) -> &'b Tensor<'b, T> {
        return &grads.grads.get(&self.id).unwrap();
    }

    pub fn backwards<'b>(&self, grad: &Tensor<'a, T>) -> Gradients<'a, T> {
        // let mut to_visit: Vec<(&Tensor<T>, HashSet<Uuid>)> = self
        //     .creators
        //     .iter()
        //     .map(|v| (*v, HashSet::from([self.id])))
        //     .collect();

        let mut to_visit: Vec<(&Tensor<T>, HashSet<Uuid>)> = vec![(self, HashSet::new())];

        let mut depends_on: HashMap<Uuid, Vec<&Tensor<T>>> = HashMap::new();

        let mut grads = Gradients {
            grads: HashMap::from([(self.id, Tensor::new(grad.data.clone()))]),
        };

        while let Some((t, h)) = to_visit.pop() {
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
            let maybe_t = depends_on.iter().find(|(_, contributors)| {
                contributors.iter().all(|t| grads.grads.contains_key(&t.id))
            }).map(|(id, _)| *id);

            let deps = depends_on.keys().collect::<Vec<&Uuid>>();
            if let Some(id) = maybe_t {
                let ts = depends_on.remove(&id).unwrap();
                let mut t_n: Option<Tensor<T>> = None;

                for t in ts {
                    if let Some(acc) = t_n {
                        let op = t.creation_op.as_ref().unwrap();
                        let grad = t.get_gradient(&grads);
                        let new_grad = op.deriv(grad);
                        t_n = Some(Tensor::new(acc.data + new_grad.data));
                    } else {
                        t_n = Some(Tensor::new(t.data.clone()));
                    }
                }

                if let Some(t) = t_n {
                    grads.grads.insert(id, t);
                } else {
                    panic!("Tensor with id {} has no upstream grads", id);
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
}

impl<'a, T> ops::Add for &'a Tensor<'a, T>
where
    T: Float,
{
    type Output = Tensor<'a, T>;

    fn add(self, _rhs: Self) -> Tensor<'a, T> {
        Tensor::<T>::new_with_creators(
            self.get_data() + _rhs.get_data(),
            vec![self, _rhs],
            Some(Box::new(Add())),
        )
    }
}

fn chapter13a() -> Result<(), std::io::Error> {
    let a = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn());
    let b = Tensor::new(array![2.0, 2.0, 2.0, 2.0, 2.0].into_dyn());
    let c = Tensor::new(array![5.0, 4.0, 3.0, 2.0, 1.0].into_dyn());
    // let d = Tensor::new(array![-1.0, -2.0, -3.0, -4.0, -5.0].into_dyn());

    let d = &a + &b;
    let e = &b + &c;
    let f = &d + &e;

    let grads = f.backwards(&Tensor::new(array![1.0, 1.0, 1.0, 1.0, 1.0].into_dyn()));

    println!("{:?}", &b.get_gradient(&grads).data);

    Ok(())
}
