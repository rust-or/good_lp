pub mod coin_cbc;

use crate::Constraint;
use crate::Variable;
use std::collections::HashMap;

#[derive(Eq, PartialEq, Clone, Copy)]
pub enum ObjectiveDirection { Maximisation, Minimisation }

#[derive(Debug, PartialEq, Clone)]
pub enum ResolutionError {
    Unbounded,
    Infeasible,
    Other(&'static str),
}

pub trait SolverModel<F> {
    type Solution: Solution<F>;
    type Error;

    fn with(self, constraint: Constraint<F>) -> Self;

    fn solve(self) -> Result<Self::Solution, Self::Error>;
}

pub trait Solution<F> {
    fn value(&self, variable: Variable<F>) -> f64;
}

impl<F, N: Into<f64> + Clone> Solution<F> for HashMap<Variable<F>, N> {
    fn value(&self, variable: Variable<F>) -> f64 {
        self[&variable].clone().into()
    }
}