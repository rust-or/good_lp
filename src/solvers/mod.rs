pub mod coin_cbc;

use crate::{Constraint, Expression};
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

    /// ## Example
    ///
    /// ```rust
    /// use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
    /// let mut vars = variables!();
    /// let a = vars.add(variable().max(1));
    /// let b = vars.add(variable().max(4));
    /// let objective = a + b;
    /// let solution = vars.maximise(objective.clone()).using(coin_cbc).solve().unwrap();
    /// assert_eq!(solution.eval(&objective), 5.);
    /// ```
    fn eval(&self, expr: &Expression<F>) -> f64 where Self: Sized {
        expr.eval_with(self)
    }
}

impl<F, N: Into<f64> + Clone> Solution<F> for HashMap<Variable<F>, N> {
    fn value(&self, variable: Variable<F>) -> f64 {
        self[&variable].clone().into()
    }
}