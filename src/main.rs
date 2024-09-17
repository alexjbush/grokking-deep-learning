use grokking_deep_learning::{Chapter, CHAPTERS};

fn print_activities(initial: bool, chapter: &Chapter) {
    if initial {
        println!("Pick an example to run:");
    } else {
        println!("Pick another example:");
    }
    for activity in chapter.activities.iter() {
        println!("{}: {}", activity.id, activity.name);
    }
    println!("(q to quit; b to go back)");
}

fn print_chapters() {
    println!("Pick a chapter to explore:");
    for chapter in CHAPTERS.iter() {
        println!("{}: {}", chapter.id, chapter.name);
    }
    println!("(q to quit)");
}

fn main() {
    let mut maybe_chapter: Option<&Chapter> = None;
    println!("Welcome to Grokking Deep Learning!\n");
    print_chapters();
    loop {
        let mut input = String::new();

        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        if input == "q" {
            break;
        }

       if let Some(chapter) = maybe_chapter {
        if input == "b" {
            maybe_chapter = None;
            print_chapters();
            continue;
        }
        let activity = chapter.activities.iter().find(|a| a.id == input);
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
                print_activities(false, chapter);
            }
            None => {
                println!("Invalid input. Try again.");
            }
        }
       } else {
        let chapter = CHAPTERS.iter().find(|a| a.id == input);
        match chapter {
            Some(chapter) => {
                maybe_chapter = Some(chapter);
                print_activities(true, chapter);
            }
            None => {
                println!("Invalid input. Try again.");
            }
       }

    }
    }
}
