use ndarray::ArrayD;
use std::ops;

use crate::{Activity, Chapter};

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Introducing automatic optimisation",
    id: "13",
};

const ACTIVITIES: [Activity; 0] = [];

trait Operation: 'static {}

struct Add();
impl Operation for Add {}
const ADD: Add = Add();

pub struct Tensor<'a, T> {
    data: ArrayD<T>,
    creators: Vec<&'a Tensor<'a, T>>,
    creation_op: Option<&'static dyn Operation>,
}

impl<'a, T> Tensor<'a, T> {
    pub fn new(data: ArrayD<T>) -> Tensor<'a, T> {
        Tensor {
            data,
            creators: vec![],
            creation_op: None,
        }
    }

    fn new_with_creators(
        data: ArrayD<T>,
        creators: Vec<&'a Tensor<'a, T>>,
        creation_op: Option<&'static dyn Operation>,
    ) -> Tensor<'a, T> {
        Tensor {
            data,
            creators,
            creation_op,
        }
    }

    pub fn get_data(&self) -> &ArrayD<T> {
        return &self.data;
    }
}

impl<'a, T> ops::Add for &'a Tensor<'a, T>
where
    &'a ArrayD<T>: ops::Add<Output = ArrayD<T>>,
{
    type Output = Tensor<'a, T>;

    fn add(self, _rhs: Self) -> Tensor<'a, T> {
        Tensor::<T>::new_with_creators(
            self.get_data() + _rhs.get_data(),
            vec![self, _rhs],
            Some(&ADD),
        )
    }
}


todo backwards