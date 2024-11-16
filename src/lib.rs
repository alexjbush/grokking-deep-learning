use chapter10::CHAPTER as CHAPTER10;
use chapter11::CHAPTER as CHAPTER11;
use chapter12::CHAPTER as CHAPTER12;
use chapter3::CHAPTER as CHAPTER3;
use chapter4::CHAPTER as CHAPTER4;
use chapter5::CHAPTER as CHAPTER5;
use chapter6::CHAPTER as CHAPTER6;
use chapter8::CHAPTER as CHAPTER8;
use chapter9::CHAPTER as CHAPTER9;
use std::io;

mod chapter10;
mod chapter11;
mod chapter12;
mod chapter3;
mod chapter4;
mod chapter5;
mod chapter6;
mod chapter8;
mod chapter9;

pub mod utils;

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

pub const CHAPTERS: [Chapter; 9] = [
    CHAPTER3, CHAPTER4, CHAPTER5, CHAPTER6, CHAPTER8, CHAPTER9, CHAPTER10, CHAPTER11, CHAPTER12,
];
