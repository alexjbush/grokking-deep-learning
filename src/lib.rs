use chapter3::CHAPTER3A;
use chapter3::CHAPTER3B;
use chapter3::CHAPTER3C;
use std::io;

pub mod chapter3;

pub struct Activity {
    pub task: fn() -> Result<(), io::Error>,
    pub name: &'static str,
    pub id: &'static str,
}

pub const ACTIVITIES: [Activity; 3] = [CHAPTER3A, CHAPTER3B, CHAPTER3C];
