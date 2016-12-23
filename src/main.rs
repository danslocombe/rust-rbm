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

pub type InPrim = u8;
pub type Input = Vec<InPrim>;
pub type Label = usize;

const LABEL_START : usize = 1;

macro_rules! from_stdin{
    ($prompt:expr, $t:ty, $default:expr) => {
        {
            println!($prompt);
            let mut input_text = String::new();
            io::stdin()
                .read_line(&mut input_text)
                .expect("failed to read from stdin");

            let trimmed = input_text.trim();
            let mut result : $t= $default;
            match trimmed.parse::<$t>() {
                Ok(i)   => result = i,
                Err(..) => println!("Failed to parse {}", input_text)
            };
            result
        }
    };
}

fn main() {
    
    let mut args = env::args();

    args.next();
    let train_data_file = match args.next(){
        Some(x) => x,
        None => panic!("No specified input data file")
    };
    let train_label_file = match args.next(){
        Some(x) => x,
        None => panic!("No specified input labels")
    };
    println!("Loading input from files...");

    let inputs = inputs_from_file(&train_data_file);
    let raw_labels : Vec<usize>= labels_from_file(&train_label_file);

    //  Parse labels
    let max_label = raw_labels.iter()
                          .map(|&x| x as usize)
                          .fold(0, |x, y| if x > y {x} else {y});

    //  TODO map with partially applied 'parse label'
    let labels : Vec<Input> = raw_labels.iter().map(|&x| {
        let mut v : Input = Vec::new();
        for i in LABEL_START..max_label{
            v.push(if i == (x as usize) {1} else {0} as InPrim);
        }
        v
    }).collect();
    println!("Done");

    let hidden_nodes = from_stdin!("Enter number of hidden nodes: ", usize, 10) as usize;

    let mut rbm = rbm::create_rbm(inputs[0].len(), hidden_nodes, max_label);
    for i in 1..100 {
        //  TODO generate minibatch
        //  ie. generate a random subset
        let batch = &inputs;
        let batch_labels = &labels;

        println!("Epoch {}", i);
        rbm.epoch(batch, batch_labels);
    }

    let sample_label_raw = from_stdin!("Enter a label to generate a sample from: ", usize, LABEL_START);
    let sample_label = parse_label(sample_label_raw, max_label as usize);
    

    println!("Sampling");
    let sample = rbm.sample(sample_label);
    println!("{:?}", sample);
}

fn parse_label(label : usize, max_label : usize) -> Input{
    let mut v : Input = Vec::new();
    for i in LABEL_START..max_label{
        v.push(if i == (label as usize) {1} else {0} as InPrim);
    }
    v
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
