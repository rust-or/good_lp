mod coin_cbc;

use crate::expression::Expression;
use crate::variable::{Constraint, ProblemVariables};
use crate::Variable;

#[derive(Eq, PartialEq, Clone, Copy)]
pub enum ObjectiveDirection { Maximisation, Minimisation }

pub trait Solution<F> {
    fn value(&self, variable: Variable<F>) -> f64;
}

#[derive(Debug, PartialEq, Clone)]
enum ResolutionError {
    Unbounded,
    Other(&'static str),
}

pub trait Solver<F> {
    type Solution: Solution<F>;
    type Error;

    fn new(
        variables: ProblemVariables<F>,
        direction: ObjectiveDirection,
        objective: Expression<F>,
    ) -> Self;

    fn with(&mut self, constraint: Constraint<F>) -> &mut Self;

    fn solve(self) -> Result<Self::Solution, Self::Error>;
}