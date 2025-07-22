use rand::Rng;
use rs_grad::{step, Var};

fn main() {
    let mut rng = rand::rng();
    let rnd_float: f64 = rng.random();
    let x: Var = rnd_float.into();
    let lr = 0.01;
    for _ in 0..1000 {
        let loss = (x.clone() - 2.0.into()) * (x.clone() - 2.0.into());
        loss.backward();
        step(&[x.clone()], lr);
        x.zero_grad()
    }
    println!("{}", x.value())
}
