use crate::Var;

pub fn step(params: &[Var], lr: f64) {
    for p in params{
        p.set_val(p.val() - p.grad() * lr)
    }
}