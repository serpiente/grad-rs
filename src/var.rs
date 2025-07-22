

use crate::ops::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Sub};
use std::ptr;
use std::rc::Rc;

struct Variable {
    val: f64,
    grad: f64,
    parents: Vec<Var>,
    op: Option<Box<dyn Op>>,
}

#[derive(Clone)]
pub struct Var(Rc<RefCell<Variable>>);

impl Var {
    pub fn new(val: f64) -> Var {
        Self(Rc::new(RefCell::new(Variable {
            val: val,
            grad: 0.0,
            parents: vec![],
            op: None,
        })))
    }

    pub fn value(&self) -> f64 {
        self.0.borrow().val
    }

    pub fn add_grad(&self, grad: f64) {
        self.0.borrow_mut().grad += grad;
    }

    fn parents(&self) -> Vec<Var> {
        self.0.borrow().parents.clone()
    }
    
    fn set_grad(&self, val: f64) {
        self.0.borrow_mut().grad = val;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    
    pub fn set_val(&self, val: f64) {
        self.0.borrow_mut().val = val;
    }

    pub fn val(&self) -> f64 {
        self.0.borrow().val
    }
    
    pub fn zero_grad(&self) {
        self.set_grad(0.0);
        for p in self.parents(){
            p.zero_grad();
        }
    }

    fn apply_op<O: Op>(o: O, vars: Vec<Var>) -> Var{
        let values:Vec<f64> =  vars.iter().map(|v| v.value()).collect();
        Self(Rc::new(RefCell::new(Variable {
            val: o.forward(&values),
            grad: 0.0,
            parents: vars,
            op: Some(Box::new(o)),
        })))
    }

    pub fn backward(&self) {
        self.set_grad(1.0);
        let nodes = self.topo_sort();
        for n in nodes.into_iter().rev() {
            let node = n.0.borrow();
            if let Some(op) = node.op.as_ref() {
                op.backward(node.grad, &n.parents());
            }
        }
    }

    fn topo_sort(&self) -> Vec<Var>{
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        self.dfs(&mut order, &mut visited);
        order
    }

    fn dfs(&self, order: &mut Vec<Var>, visited: &mut HashSet<Var>) {
        if visited.insert(self.clone()) {
            for parent in self.parents() {
                parent.dfs(order, visited);
            }
            order.push(self.clone());
        }
    }
    
}

impl PartialEq<Self> for Var {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(Rc::as_ptr(&self.0), Rc::as_ptr(&other.0))
    }
}

impl Eq for Var {}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}


impl Into<Var> for f64 {
    fn into(self) -> Var {
        Var::new(self)
    }
}

impl Add for Var {
    type Output = Var;
    fn add(self, other: Var) -> Var {
        Var::apply_op(AddOp, vec![self, other])
    }
}

impl Sub for Var {
    type Output = Var;
    fn sub(self, other: Var) -> Var {
        Var::apply_op(AddOp, vec![self, other])
    }
}

impl Mul for Var {
    type Output = Var;
    fn mul(self, other: Self) -> Self::Output {
        Var::apply_op(MulOp, vec![self, other])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into() {
        let var: Var = 3.0.into();
        assert_eq!(var.0.borrow().val, 3.0);
    }

    fn test_add(){
        let lhs: Var = 3.0.into();
        let rhs: Var = 3.0.into();
        let r = lhs + rhs;
        assert_eq!(r.value(), 6.0);
    }

    #[test]
    fn test_add_sets_parents() {
        let x: Var = 2.0.into();
        let y: Var = 3.0.into();
        let z = x + y;

        let parents = &z.0.borrow().parents;
        assert_eq!(parents.len(), 2);
        assert_eq!(parents[0].0.borrow().val, 2.0);
        assert_eq!(parents[1].0.borrow().val, 3.0);
    }

    #[test]
    fn test_backwards_sum() {
        let x: Var = 2.0.into();
        let y: Var = 3.0.into();
        let z = x.clone() + y.clone();
        z.backward();
        assert_eq!(x.grad(), 1.0);
        assert_eq!(y.grad(), 1.0);
    }
}
