extern crate rand;
use rand::Rng;

struct Neuron {   
    state  : i8,
    bias   : f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn field(&self, connections : &Vec<Neuron>) -> f32 {
        let inputs : Vec<f32> = connections.iter().zip(self.weights.iter())
                                                  .map(|(n, &y)|(n.state as f32)*y)
                                                  .collect();
        self.bias + inputs.iter().fold(0.0, |sum, x| sum + x)
    }
}

fn detUpdate(n : &Neuron, connections : &Vec<Neuron>) -> Neuron {
    let new_state = if n.field(connections) >= 0.0 {1} else {-1};
    Neuron {state : new_state, bias : n.bias, weights : n.weights.clone()}
}

fn stochUpdate(n : &Neuron, connections : &Vec<Neuron>, t : f32) -> Neuron {
    let ht = -n.field(connections) / t;
    let p = 1.0 / (1.0 + ht.exp());
    let rand : f32 = rand::thread_rng().gen_range(0.0, 1.0);
    let new_state = if rand < p {1} else {0};
    Neuron {state : new_state, bias : n.bias, weights : n.weights.clone()}
}

struct RBM {
    size   : i32,
    visible: Vec<Neuron>,
    hidden : Vec<Neuron>,
}

impl RBM {
    fn update<F>(&self, update : F) -> RBM where
        F: Fn(&Neuron, &Vec<Neuron>) -> Neuron{

        let mut new_vis : Vec<Neuron> = Vec::new();
        for n in &self.visible {
            let x = update(&n, &self.hidden); 
            new_vis.push(x);
        }
        let mut new_hid : Vec<Neuron> = Vec::new();
        for n in &self.hidden {
            let x = update(&n, &self.visible); 
            new_hid.push(x);
        }
        RBM {size : self.size, visible: new_vis, hidden : new_hid}
    }
}
