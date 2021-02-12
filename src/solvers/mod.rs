pub mod coin_cbc;

use crate::Constraint;
use crate::Variable;

#[derive(Eq, PartialEq, Clone, Copy)]
pub enum ObjectiveDirection { Maximisation, Minimisation }

#[derive(Debug, PartialEq, Clone)]
pub enum ResolutionError {
    Unbounded,
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
