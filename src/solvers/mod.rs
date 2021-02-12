#[cfg(feature = "coin_cbc")]
pub mod coin_cbc;

use crate::Variable;
use crate::{Constraint, Expression};
use std::collections::HashMap;

/// Whether to search for the variable values that give the highest
/// or the lowest value of the objective function.
#[derive(Eq, PartialEq, Clone, Copy)]
pub enum ObjectiveDirection {
    Maximisation,
    Minimisation,
}

/// Represents an error that occurred when solving a problem
#[derive(Debug, PartialEq, Clone)]
pub enum ResolutionError {
    /// The problem is [unbounded](https://www.matem.unam.mx/~omar/math340/unbounded.html).
    /// It doesn't have a finite optimal values for its variables.
    /// The objective can be made infinitely large without violating any constraints.
    Unbounded,
    ///  There exists no solution that satisfies all of the constraints
    Infeasible,
    /// Another error occurred
    Other(&'static str),
}

/// A solver's own representation of a model, to which constraints can be added.
pub trait SolverModel<F> {
    /// The type of the solution to the problem
    type Solution: Solution<F>;
    /// The error that can occur while solving the problem
    type Error;

    /// Takes a model and adds a constraint to it
    fn with(self, constraint: Constraint<F>) -> Self;

    /// Find the solution for the problem being modeled
    fn solve(self) -> Result<Self::Solution, Self::Error>;
}

/// A problem solution
pub trait Solution<F> {
    /// Get the optimal value of a variable of the problem
    fn value(&self, variable: Variable<F>) -> f64;

    /// ## Example
    ///
    /// ```rust
    /// # #[cfg(feature = "coin_cbc")] {
    /// use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
    /// let mut vars = variables!();
    /// let a = vars.add(variable().max(1));
    /// let b = vars.add(variable().max(4));
    /// let objective = a + b;
    /// let solution = vars.maximise(objective.clone()).using(coin_cbc).solve().unwrap();
    /// assert_eq!(solution.eval(&objective), 5.);
    /// # }
    /// ```
    fn eval(&self, expr: &Expression<F>) -> f64
    where
        Self: Sized,
    {
        expr.eval_with(self)
    }
}

impl<F, N: Into<f64> + Clone> Solution<F> for HashMap<Variable<F>, N> {
    fn value(&self, variable: Variable<F>) -> f64 {
        self[&variable].clone().into()
    }
}
