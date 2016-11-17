struct Neuron {   
    state  : i8,
    bias   : f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn update(&self, connections : &Vec<Neuron>) -> Neuron {
        let inputs : Vec<f32> = connections.iter().zip(self.weights.iter())
                                                  .map(|(n, &y)|(n.state as f32)*y)
                                                  .collect();
        let sum : f32 = self.bias + inputs.iter().fold(0.0, |sum, x| sum + x);
        let newstate = if sum >= 0.0 {1} else {-1};
        Neuron {state : newstate, bias : self.bias, weights : self.weights.clone()}
    }
}

struct RBM {
    size   : i32,
    visible: Vec<Neuron>,
    hidden : Vec<Neuron>,
}

impl RBM {
    fn update(&self) -> RBM {
        let mut new_vis : Vec<Neuron> = Vec::new();
        for n in &self.visible {
            let x = n.update(&self.hidden); 
            new_vis.push(x);
        }
        let mut new_hid : Vec<Neuron> = Vec::new();
        for n in &self.hidden {
            let x = n.update(&self.visible); 
            new_hid.push(x);
        }
        RBM {size : self.size, visible: new_vis, hidden : new_hid}
    }
}

fn main() {
    println!("Hello, world!");
}
