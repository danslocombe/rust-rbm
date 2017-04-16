use rand::{thread_rng, Rng};

use Input;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

struct LVProbs {
    visible: Vector<f32>,
    labels: Vector<f32>,
}

pub struct RBM {
    visible_size: usize,
    hidden_size: usize,
    labels: usize,
    weights: Matrix<f32>,
    vbias: Vector<f32>,
    hbias: Vector<f32>,

    //  Number of monte carlo marcov chain iterations to use when sampling
    sample_mcmc: usize,
}

impl RBM {
    fn prop_hidden(&self, hidden: &Vector<f32>) -> Vector<f32> {
        let vs = &(self.weights) * hidden + &(self.vbias);
        Vector::from(vs.iter()
            .map(|field| sigmoid(field))
            .collect::<Vec<f32>>())
    }
    fn prop_visible(&self, visible: &Vector<f32>) -> Vector<f32> {
        let vs = &(self.weights.transpose()) * visible + &(self.hbias);
        Vector::from(vs.iter()
            .map(|field| sigmoid(field))
            .collect::<Vec<f32>>())
    }

    pub fn epoch(&mut self, batch: &[Input], labels: &[Input]) {


        //  Construct delta values
        let mut d_weights: Matrix<f32> = Matrix::zeros(self.labels + self.visible_size,
                                                       self.hidden_size);
        let mut d_vbiases: Vector<f32> = Vector::zeros(self.labels + self.visible_size);
        let mut d_hbiases: Vector<f32> = Vector::zeros(self.hidden_size);

        let batch_labels = batch.iter().zip(labels.iter());

        for (example, label) in batch_labels {

            //  Convert example and label into one visible layer Vector
            let v0 = Vector::from(example.iter()
                .chain(label.iter())
                .map(|x| *x as f32)
                .collect::<Vec<_>>());

            //  Contrastive divergence, for k = 1
            let h0 = self.prop_visible(&v0);
            let vk = self.prop_hidden(&h0);
            let hk = self.prop_visible(&vk);

            //  Update delta weights
            for i in 0..self.visible_size {
                for j in 0..self.hidden_size {
                    d_weights[[i, j]] += h0[j] * v0[i] - hk[j] * vk[i];
                }
            }

            //  Update delta weights for visible layer
            for i in 1..self.visible_size {
                d_vbiases[i] += v0[i] - vk[i];
            }

            //  Update delta weights for hidden layer
            for j in 1..self.hidden_size {
                d_hbiases[j] += h0[j] - hk[j];
            }

        }

        let learning_rate = 0.05;
        let ll2 = 0.01;

        //  TODO Add Momentum
        let v = 0.01;
        //

        let batch_size = batch.len() as f32;
        let lr = learning_rate / batch_size;

        for i in 1..self.visible_size {
            for j in 1..self.hidden_size {
                self.weights[[i, j]] += lr * d_weights[[i, j]] - ll2 * self.weights[[i, j]];
            }
        }

        for i in 1..self.visible_size {
            self.vbias[i] += lr * d_vbiases[i] - ll2 * self.vbias[i];
        }

        for j in 1..self.hidden_size {
            self.hbias[j] += lr * d_hbiases[j] - ll2 * self.hbias[j];
        }
    }

    pub fn sample(&self, clamped_labels: Input) -> Vector<f32> {
        let labels = clamped_labels.iter().map(|x| *x as f32).collect::<Vec<_>>();
        let input: Vec<f32> = rand_vec(self.visible_size)
            .iter()
            .cloned()
            .chain(labels.iter().cloned())
            .collect();

        let mut v = Vector::from(input);
        let mut h = self.prop_visible(&v);

        for _ in 1..self.sample_mcmc {
            h = self.prop_visible(&v);
            v = Vector::from(self.prop_hidden(&h)
                .iter()
                .take(self.visible_size)
                .cloned()
                .chain(labels.iter().cloned())
                .collect::<Vec<_>>());
        }
        Vector::from(v.iter().take(self.visible_size).cloned().collect::<Vec<_>>())
    }
}

fn rand_vec(size: usize) -> Vec<f32> {
    let mut v = Vec::new();
    for _ in 0..(size) {
        v.push(thread_rng().gen_range(0.0, 1.0));
    }
    v
}

fn step_update(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}


pub fn randomise_weight_matrix(vsize: usize, hsize: usize) -> Matrix<f32> {
    let mut ws_values = Vec::new();
    for _ in 0..(vsize * hsize) {
        ws_values.push(thread_rng().gen_range(0.0, 1.0));
    }

    Matrix::new(vsize, hsize, ws_values)
}


pub fn new(vsize: usize, hsize: usize, labels: usize) -> RBM {

    let mut vb = Vec::new();
    for _ in 0..vsize + labels {
        vb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    let mut hb = Vec::new();
    for _ in 0..hsize {
        hb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    RBM {
        visible_size: vsize,
        hidden_size: hsize,
        labels: labels,
        weights: randomise_weight_matrix(vsize + labels, hsize),
        vbias: Vector::new(vb),
        hbias: Vector::new(hb),
        sample_mcmc: 32,
    }
}
