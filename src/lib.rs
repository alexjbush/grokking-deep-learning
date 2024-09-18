use chapter3::CHAPTER as CHAPTER3;
use chapter4::CHAPTER as CHAPTER4;
use chapter5::CHAPTER as CHAPTER5;
use std::io;

mod chapter3;
mod chapter4;
mod chapter5;

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

pub const CHAPTERS: [Chapter; 3] = [CHAPTER3, CHAPTER4, CHAPTER5];
