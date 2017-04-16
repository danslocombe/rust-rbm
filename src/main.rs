extern crate csv;
extern crate rulinalg;
extern crate rand;
extern crate kronecker;

mod rbm;

use std::env;
use std::io;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::result;
use std::fmt;
use kronecker::delta;

pub struct RBMError(String);
impl fmt::Display for RBMError {
    fn fmt(&self, formatter : &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        self.0.fmt(formatter)
    }
}
impl<'a> From<&'a str> for RBMError {
    fn from(s : &'a str) -> Self {
        RBMError(s.to_owned())
    }
}

pub type Result<T> = result::Result<T, RBMError>;

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
    try_main().map_err(|e| println!("Error: {}", e));
}

fn try_main() -> Result<()> {
    let mut args = env::args();

    args.next();
    let train_data_file = args.next().ok_or("No specified input data file")?;
    let train_label_file = args.next().ok_or("No specified input labels")?;    

    println!("Loading inputs from files...");
    let inputs = inputs_from_file(&train_data_file)?;

    println!("Loading inputs from files...");
    let raw_labels : Vec<usize>= labels_from_file(&train_label_file)?;

    //  Parse labels
    let max_label = raw_labels.iter()
                              .map(|&x| x as Label)
                              .fold(0, |x, y| if x > y {x} else {y});

    //  TODO map with partially applied 'parse label'
    let labels : Vec<Input> = raw_labels.iter().map(|&x| {
        let mut v : Input = Vec::new();
        for i in LABEL_START..max_label + 1{
            v.push(delta(&i, &(x as usize)));
        }
        v
    }).collect();
    println!("Done");

    let hidden_nodes = from_stdin!("Enter number of hidden nodes: ", usize, 10) as usize;

    let mut rbm = rbm::new(inputs[0].len(), hidden_nodes, max_label);
    for i in 1..100 {
        //  TODO generate minibatch
        //  ie. generate a random subset
        let batch = &inputs;
        let batch_labels = &labels;

        rbm.epoch(batch, batch_labels);
    }

    let sample_label_raw = from_stdin!("Enter a label to generate a sample from: ", usize, LABEL_START);
    let sample_label = parse_label(sample_label_raw, max_label as usize);
    

    println!("Sampling");
    let sample = rbm.sample(sample_label);
    println!("{:?}", sample);

    Ok(())
}

fn parse_label(label : usize, max_label : usize) -> Input{
    let mut v : Input = Vec::new();
    for i in LABEL_START..max_label + 1 {
        v.push(if i == (label as usize) {1} else {0} as InPrim);
    }
    v
}

fn inputs_from_file(filename : &String) -> Result<Vec<Input>> {
    let f = read_file(&filename)?;
    let mut rdr = csv::Reader::from_string(f).has_headers(false);

    //  Convert strings to uint8s
    //  Implicitly convert Vec<Result<_>> to Result<Vec<_>>
    rdr.records().map(|r| {
        r.unwrap().iter().map(|x|{
            x.parse::<u8>().map_err(|why| {
                RBMError(format!("Malformed CSV file, {}\n{}", filename, why))
            })
        }).collect()
    }).collect()
}

fn labels_from_file(filename : &String) -> Result<Vec<Label>> {
    let f = read_file(&filename)?;
    f.split("\n").filter(|x| !x.is_empty()).map(|x|{
        x.parse::<Label>().map_err(|why| {
            RBMError(format!("Malformed label file, {}\n{}", filename, why))
        })
    }).collect()
}

fn read_file(s : &String) -> Result<String> {
    // Create a path to the desired file
    let path = Path::new(s);
    let display = path.display();

    // Open the path in read-only mode
    let mut file = File::open(&path).map_err(|why| {
        RBMError(format!("Couldn't open {}: {}", display,
                                                   why.description()))
    })?;

    // Read the file contents into a string
    let mut s = String::new();
    file.read_to_string(&mut s).map_err(|why| {
        RBMError(format!("couldn't read {}: {}", display,
                                                   why.description()))
    });
    Ok(s)
}
