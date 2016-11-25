use rand::{thread_rng, Rng};

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
    fn propHidden (self, hidden : &Vector<f32>) -> Vector<f32> {
        let mut vs = self.weights * hidden + self.vbias;
        vs.iter_mut().map(|field| stepUpdate(field.to_owned()));
        vs
    }
    fn propVisible (self, visible : &Vector<f32>) -> Vector<f32> {
        let mut hs = self.weights.transpose() * visible + self.hbias;
        hs.iter_mut().map(|field| stepUpdate(field.to_owned()));
        hs
    }
}

fn stepUpdate(x : f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
}

pub fn randomiseWeightMatrix (vsize : usize, hsize : usize) -> Matrix<f32> {
    let mut ws_values = Vec::new();
    for i in 0..(vsize * hsize) {
        ws_values.push(thread_rng().gen_range(0.0, 1.0));
    }

    Matrix::new(vsize, hsize, ws_values)
}

pub fn createRBM (vsize : usize, hsize : usize, wm : &mut Matrix<f32>) -> RBM {

    let mut vb = Vec::new();
    for i in 0..vsize {
        vb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    let mut hb = Vec::new();
    for i in 0..hsize {
        vb.push(thread_rng().gen_range(-1.0, 1.0));
    }

    RBM {
        visible_size : vsize,
        hidden_size  : hsize,
        weights      : randomiseWeightMatrix(vsize, hsize),
        vbias        : Vector::new(vb),
        hbias        : Vector::new(hb),
    }
}
