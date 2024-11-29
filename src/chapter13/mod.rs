use crate::{Activity, Chapter};
use ndarray::array;
use ndarray::ArrayD;
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

const ACTIVITIES: [Activity; 2] = [CHAPTER13A, CHAPTER13B];

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
struct Expand();
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
            .map(|t| (t.id, Tensor::new(grad.data.clone())))
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
        HashMap::from([(creators[0].id, Tensor::new(-&grad.data))])
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
            (creators[0].id, Tensor::new(grad.data.clone())),
            (creators[1].id, Tensor::new(-&grad.data)),
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
            (creators[0].id, Tensor::new(&grad.data * &creators[1].data)),
            (creators[1].id, Tensor::new(&grad.data * &creators[0].data)),
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
        todo!();
        // let act = creators[0];
        // let weights = creators[1];
        // HashMap::from([
        //     (creators[0].id, Tensor::new(&grad.data * &creators[1].data)),
        //     (creators[1].id, Tensor::new(&grad.data * &creators[0].data)),
        // ])
    }
}

impl Operation for Sum {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        todo!();
        // HashMap::from([(creators[0].id, Tensor::new(grad.data.t().to_owned()))])
    }
}

impl Operation for Expand {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        todo!()
    }
}

impl Operation for Transpose {
    fn deriv(
        &self,
        grad: &Tensor<f64>,
        creators: &Vec<&Tensor<f64>>,
    ) -> HashMap<Uuid, Tensor<f64>> {
        assert!(creators.len() == 1);
        HashMap::from([(creators[0].id, Tensor::new(grad.data.t().to_owned()))])
    }
}

pub struct Gradients<'a> {
    grads: HashMap<Uuid, Tensor<'a, f64>>,
}

pub struct Tensor<'a, T> {
    id: Uuid,
    data: ArrayD<T>,
    creators: Vec<&'a Tensor<'a, T>>,
    creation_op: Option<Box<dyn Operation>>,
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
        creation_op: Option<Box<dyn Operation>>,
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
            grads: HashMap::from([(self.id, Tensor::new(grad.data.clone()))]),
        };

        let mut id_to_tensor: HashMap<Uuid, &Tensor<f64>> = HashMap::new();

        while let Some((t, h)) = to_visit.pop() {
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
                            grads.grads.insert(id, Tensor::new(&gg.data + g.data));
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
        Tensor::new_with_creators(
            self.get_data().sum_axis(ndarray::Axis(dim)),
            vec![self],
            Some(Box::new(Sum { dim })),
        )
    }

    pub fn transpose(&'a self) -> Tensor<'a, f64> {
        Tensor::new_with_creators(
            self.get_data().t().to_owned(),
            vec![self],
            Some(Box::new(Transpose())),
        )
    }

    pub fn expand(&'a self, dim: usize, copies: usize) -> Tensor<'a, f64> {
        this doesn;t work
    }
}

impl<'a> ops::Add for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn add(self, _rhs: Self) -> Self::Output {
        Tensor::new_with_creators(
            self.get_data() + _rhs.get_data(),
            vec![self, _rhs],
            Some(Box::new(Add())),
        )
    }
}

impl<'a> ops::Neg for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn neg(self) -> Self::Output {
        Tensor::new_with_creators(-(self.get_data()), vec![self], Some(Box::new(Neg())))
    }
}

impl<'a> ops::Sub for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn sub(self, _rhs: Self) -> Self::Output {
        Tensor::new_with_creators(
            self.get_data() - _rhs.get_data(),
            vec![self, _rhs],
            Some(Box::new(Sub())),
        )
    }
}

impl<'a> ops::Mul for &'a Tensor<'a, f64> {
    type Output = Tensor<'a, f64>;

    fn mul(self, _rhs: Self) -> Self::Output {
        Tensor::new_with_creators(
            self.get_data() * _rhs.get_data(),
            vec![self, _rhs],
            Some(Box::new(Mul())),
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

fn chapter13b() -> Result<(), std::io::Error> {
    let a = Tensor::new(array![1.0, 2.0, 3.0, 4.0, 5.0].into_dyn());
    let b = Tensor::new(array![2.0, 2.0, 2.0, 2.0, 2.0].into_dyn());
    let c = Tensor::new(array![5.0, 4.0, 3.0, 2.0, 1.0].into_dyn());
    // let d = Tensor::new(array![-1.0, -2.0, -3.0, -4.0, -5.0].into_dyn());

    let negb1 = -&b;
    let negb2 = -&b;
    let d = &a + &negb1;
    let e = &negb2 + &c;
    let f = &d + &e;

    let grads = f.backwards(&Tensor::new(array![1.0, 1.0, 1.0, 1.0, 1.0].into_dyn()));

    println!("{:?}", &b.get_gradient(&grads).data);

    Ok(())
}
