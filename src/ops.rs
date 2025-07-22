
use crate::var::Var;
pub trait Op: 'static {
    fn forward(&self, inputs: &[f64]) -> f64;
    fn backward(&self, grad: f64, parents: &[Var]);
}

pub struct AddOp;
pub struct MulOp;


impl Op for AddOp {
    fn forward(&self, inputs: &[f64]) -> f64 {
        inputs[0] + inputs[1]
    }
    fn backward(&self, grad: f64, parents: &[Var]) {
        for p in parents {
            p.add_grad(grad);
        }
    }
}

impl Op for MulOp {
    fn forward(&self, inputs: &[f64]) -> f64 {
        inputs[0] * inputs[1]
    }

    fn backward(&self, grad: f64, parents: &[Var]) {
        let x = parents[0].value();
        let y = parents[1].value();
        parents[0].add_grad(y * grad);
        parents[1].add_grad(x * grad);
    }
}
