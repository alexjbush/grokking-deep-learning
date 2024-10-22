use std::{fs::File, path::{Path, PathBuf}, thread};
use std::io::{Write};

use std::fs;
use curl::easy::Easy;
use pbr::ProgressBar;


#[cfg(target_family = "unix")]
use std::os::unix::fs::MetadataExt;
#[cfg(target_family = "windows")]
use std::os::windows::fs::MetadataExt;

#[cfg(target_family = "unix")]
fn file_size(meta: &dyn MetadataExt) -> usize {
    meta.size() as usize
}

#[cfg(target_family = "windows")]
fn file_size(meta: &MetadataExt) -> usize {
    meta.file_size() as usize
}

pub static BASE_PATH: &str = ".data/";
pub static GROKKING_BASE_URL: &str = "https://github.com/iamtrask/Grokking-Deep-Learning/raw/refs/heads/master";

pub fn download_files(
    base_url: &str,
    base_path: &str,
    files_to_download: &[(&str, usize)]
) -> Result<(), String> {
    let download_dir = PathBuf::from(base_path);
    if !download_dir.exists() {
        log::info!(
            "Download directory {} does not exists. Creating....",
            download_dir.display()
        );
        fs::create_dir_all(&download_dir).or_else(|e| {
            Err(format!(
                "Failed to create directory {:?}: {:?}",
                download_dir, e
            ))
        })?;
    }

    for &(archive, size) in files_to_download {
        log::info!("Attempting to download and extract {}...", archive);
        download(base_url, archive, size, &download_dir)?;
    }
    Ok(())
}

fn download(
    base_url: &str,
    archive: &str,
    full_size: usize,
    download_dir: &Path,
) -> Result<(), String> {
    let mut easy = Easy::new();
    let url = Path::new(base_url).join(archive);
    let file_name = download_dir.to_str().unwrap().to_owned() + archive; //.clone();
    if Path::new(&file_name).exists() {
        log::info!(
            "  File {:?} already exists, skipping downloading.",
            file_name
        );
    } else {
        log::info!(
            "- Downloading from file from {} and saving to file as: {}",
            url.to_str().unwrap(),
            file_name
        );

        let mut file = File::create(file_name.clone()).unwrap();
        let pb_thread = thread::spawn(move || {
            let mut pb = ProgressBar::new(full_size.try_into().unwrap());
            pb.format("╢=> ╟");

            let mut current_size = 0;
            while current_size < full_size {
                let meta = fs::metadata(file_name.clone())
                    .expect(&format!("Couldn't get metadata on {:?}", file_name));

                current_size = file_size(&meta);

                pb.set(current_size.try_into().unwrap());
                thread::sleep_ms(10);
            }
            pb.finish_println(" ");
        });
        easy.url(&url.to_str().unwrap()).unwrap();
        easy.get(true).unwrap();
        easy.max_redirections(10).unwrap();
        easy.follow_location(true).unwrap();
        easy.write_function(move |data| {
            file.write_all(data).unwrap();
            Ok(data.len())
        })
        .unwrap();
        easy.perform().unwrap();
        pb_thread.join().unwrap();
    }

    Ok(())
}
