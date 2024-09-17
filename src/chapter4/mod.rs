use crate::{Activity, Chapter};

const ACTIVITIES: [Activity; 2] = [CHAPTER4A, CHAPTER4B];

pub const CHAPTER: Chapter = Chapter {
    activities: &ACTIVITIES,
    name: "Introduction to neural learning",
    id: "4",
};

const CHAPTER4A: Activity = Activity {
    task: chapter4a,
    name: "Hot and cold learning",
    id: "4a",
};

fn chapter4a() -> Result<(), std::io::Error> {
    let mut weight: f64 = 0.5;
    let input: f64 = 0.5;
    let goal_prediction: f64 = 0.8;

    let step_amount = 0.001;
    let tolerance: f64 = 0.000000001;

    for iteration in  0..2000 {

        let prediction = input * weight;
        let error = (prediction - goal_prediction).powi(2);

        println!("Iteration: {}, Error: {}, Prediction: {}", iteration, error, prediction);

        let up_prediction = input * (weight + step_amount);
        let up_error = (goal_prediction - up_prediction).powi(2);

        let down_prediction = input * (weight - step_amount);
        let down_error = (goal_prediction - down_prediction).powi(2);

        if error < tolerance {
            break;
        }
        else if down_error < up_error {
            weight = weight - step_amount;
        } else if down_error > up_error {
            weight = weight + step_amount;
        }

    }
    Ok(())
}

const CHAPTER4B: Activity = Activity {
    task: chapter4b,
    name: "Direction and amount",
    id: "4b",
};

fn chapter4b() -> Result<(), std::io::Error> {
    let mut weight: f64 = 0.5;
    let input: f64 = 0.5;
    let goal_prediction: f64 = 0.8;

    let tolerance: f64 = 0.000000001;

    for iteration in  0..2000 {

        let prediction = input * weight;
        let error = (prediction - goal_prediction).powi(2);

        println!("Iteration: {}, Error: {}, Prediction: {}", iteration, error, prediction);

        if error < tolerance {
            break;
        }
        else {
            let direction_and_amount = (prediction - goal_prediction) * input;
            weight = weight - direction_and_amount;
        }

    }
    Ok(())
}