extern crate csv;
extern crate rulinalg;
extern crate rand;

mod rbm;

use std::env;
use std::io;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

pub type Input = Vec<u8>;
pub type Label = u32;

pub fn test() {
}

fn main() {
    
    let mut args = env::args();

    args.next();
    let train_data_file = match args.next(){
        Some(x) => x,
        None => panic!("No specified input data file")
    };
    /*
    let train_label_file = match args.next(){
        Some(x) => x,
        None => panic!("No specified input labels")
    };
    */
    println!("Loading input from files...");
    let inputs = inputs_from_file(&train_data_file);
    //let labels = labels_from_file(&train_label_file);
    println!("Done");

    println!("Enter number of hidden nodes: ");
    let mut input_text = String::new();
    io::stdin()
        .read_line(&mut input_text)
        .expect("failed to read from stdin");

    let trimmed = input_text.trim();
    let mut hidden_nodes = 10;
    match trimmed.parse::<usize>() {
        Ok(i)   => hidden_nodes = i,
        Err(..) => println!("Failed to parse {}", input_text)
    };
    let mut rbm = rbm::create_rbm(inputs[0].len(), hidden_nodes);
    for i in 1..100 {
        println!("Epoch {}", i);
        rbm.epoch(&inputs);
    }

    println!("Sampling");
    let sample = rbm.sample();
    println!("{:?}", sample);
}

fn inputs_from_file(filename : &String) -> Vec<Input> {
    let f = read_file(&filename);
    let mut rdr = csv::Reader::from_string(f).has_headers(false);

    //  Convert strings to uint8s
    rdr.records().map(|r| {
        r.unwrap().iter().map(|x|{
            match x.parse::<u8>(){
                Err(why) => panic!("Error in csv file, {}\n{}", filename, why),
                Ok(a) => a,
            }
        }).collect()
    }).collect()
}

fn labels_from_file(filename : &String) -> Vec<Label> {
    let f = read_file(&filename);
    f.split("\n").map(|x|{
        match x.parse::<Label>(){
            Err(why) => panic!("Error in label file, {}\n{}", filename, why),
            Ok(a) => a,
        }
    }).collect::<Vec<Label>>()
}

fn read_file(s : &String) -> String {
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
