use grokking_deep_learning::ACTIVITIES;

fn print_activities(initial: bool) {
    if initial {
        println!("Pick an example to run:");
    } else {
        println!("Pick another example:");
    }
    for activity in ACTIVITIES.iter() {
        println!("{}: {}", activity.id, activity.name);
    }
    println!("(q to quit)");
}

fn main() {
    println!("Welcome to Grokking Deep Learning!\n");
    print_activities(true);
    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        if input == "q" {
            break;
        }
        let activity = ACTIVITIES.iter().find(|a| a.id == input);
        match activity {
            Some(activity) => {
                println!("Running {}...", activity.name);
                let r = (activity.task)();
                match r {
                    Ok(_) => {
                        println!("Done!\n");
                    }
                    Err(e) => {
                        println!("Error: {:?}", e);
                    }
                }
                print_activities(false);
            }
            None => {
                println!("Invalid input. Try again.");
            }
        }
    }
}
