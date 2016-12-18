use rand::{thread_rng, Rng};

use Input;
use Label;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

pub struct RBM{
    visible_size : usize, 
    hidden_size  : usize, 
    weights      : Matrix<f32>,
    vbias        : Vector<f32>,
    hbias        : Vector<f32>,
}

impl RBM{
    fn prop_hidden (&self, hidden : &Vector<f32>) -> Vector<f32> {
        let vs = &(self.weights) * hidden + &(self.vbias);
        Vector::from(
            vs.iter().map(
                |field| sigmoid(field.to_owned())).
            collect::<Vec<f32>>()
        )
    }
    fn prop_visible (&self, visible : &Vector<f32>) -> Vector<f32> {
        let vs = &(self.weights.transpose()) * visible + &(self.hbias);
        Vector::from(
            vs.iter().map(
                |field| sigmoid(field.to_owned())).
            collect::<Vec<f32>>()
        )
    }

    pub fn epoch (&mut self, batch : &Vec<Input>){

        let mut d_weights : Matrix<f32> = Matrix::zeros(self.visible_size, self.hidden_size);
        let mut d_vbiases : Vector<f32> = Vector::zeros(self.visible_size);
        let mut d_hbiases : Vector<f32> = Vector::zeros(self.hidden_size);


        for v_batch in batch{

            //  Convert to Vec of f32
            let v_batch1 = v_batch.iter().map(|x| x.to_owned() as f32).collect::<Vec<f32>>();

            //  Contrastive divergence, for now k = 1
            let v0 = Vector::from(v_batch1);
            let h0 = self.prop_visible(&v0);
            let vk = self.prop_hidden(&h0);
            let hk = self.prop_visible(&vk);


            for i in 1..self.visible_size{
                for j in 1..self.hidden_size{
                    d_weights[[i, j]] += h0[j] * v0[i] - hk[j] * vk[i];
                    d_vbiases[i] += v0[i] - vk[i];
                    d_hbiases[j] += h0[j] - hk[j];
                }
            }

        }

        let learning_rate = 0.05;
        let ll2 = 0.01;
        let v = 0.01;

        let batch_size = batch.len() as f32;
        let lr = learning_rate / batch_size;

        for i in 1..self.visible_size{
            for j in 1..self.hidden_size{
                //  TODO adjust based on d_weights derivitive
                self.weights[[i, j]] += lr * d_weights[[i, j]] - ll2 * self.weights[[i, j]];
                self.vbias[i] += lr * d_vbiases[i] - ll2 * self.vbias[i];
                self.hbias[j] += lr * d_hbiases[j] - ll2 * self.hbias[j];
            }
        }
    }

    pub fn sample(&self) -> Vector<f32> {
        //  Create random vector
        let mut v = Vector::from(rand_vec(self.visible_size));
        let mut h = self.prop_visible(&v);

        let mcmcs = 32;
        for i in 1..mcmcs {
            h = self.prop_visible(&v);
            v = self.prop_hidden(&h);
        }
        v
    }
}

fn rand_vec(size : usize) -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..(size) {
        v.push(thread_rng().gen_range(0.0, 1.0));
    }
    v
}

fn step_update(x : f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
}

fn sigmoid (x : f32) -> f32{
    1.0 / (1.0 + (-x).exp())
}


pub fn randomise_weight_matrix(vsize : usize, hsize : usize) -> Matrix<f32> {
    let mut ws_values = Vec::new();
    for i in 0..(vsize * hsize) {
        ws_values.push(thread_rng().gen_range(0.0, 1.0));
    }

    Matrix::new(vsize, hsize, ws_values)
}


pub fn create_rbm(vsize : usize, hsize : usize) -> RBM {

    let mut vb = Vec::new();
    for i in 0..vsize {
        vb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    let mut hb = Vec::new();
    for i in 0..hsize {
        hb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    RBM {
        visible_size : vsize,
        hidden_size  : hsize,
        weights      : randomise_weight_matrix(vsize, hsize),
        vbias        : Vector::new(vb),
        hbias        : Vector::new(hb),
    }
}
