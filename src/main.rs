extern crate csv;
extern crate rulinalg;
extern crate rand;

mod rbm;

use std::env;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

type Input = Vec<u8>;

fn main() {
    
    let mut args = env::args();

    args.next();
    let infile = match args.next(){
        Some(x) => x,
        None => panic!("")
    };
    println!("Loading input from files...");
    let inputs = inputsFromFile(&infile);
    println!("Done");
}

fn inputsFromFile(filename : &String) -> Vec<Input> {
    let f = readfile(&filename);
    let mut rdr = csv::Reader::from_string(f).has_headers(false);

    //  Convert strings to uint8s
    rdr.records().map(|r| {
        r.unwrap().iter().map(|x|{
            match x.parse::<u8>(){
                Err(why) => panic!("Error in csv file, {}", filename),
                Ok(a) => a,
            }
        }).collect()
    }).collect()
}

fn readfile(s : &String) -> String {
    // Create a path to the desired file
    let path = Path::new(s);
    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!("couldn't open {}: {}", display,
                                                   why.description()),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display,
                                                   why.description()),
        Ok(_) => ()
    }
    s
    // `file` goes out of scope, and the "hello.txt" file gets closed
}
