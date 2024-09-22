use chapter3::CHAPTER as CHAPTER3;
use chapter4::CHAPTER as CHAPTER4;
use chapter5::CHAPTER as CHAPTER5;
use chapter6::CHAPTER as CHAPTER6;
use std::io;

mod chapter3;
mod chapter4;
mod chapter5;
mod chapter6;

pub struct Activity {
    pub task: fn() -> Result<(), io::Error>,
    pub name: &'static str,
    pub id: &'static str,
}

pub struct Chapter {
    pub activities: &'static [Activity],
    pub name: &'static str,
    pub id: &'static str,
}

pub const CHAPTERS: [Chapter; 4] = [CHAPTER3, CHAPTER4, CHAPTER5, CHAPTER6];
