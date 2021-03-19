//! Included solvers that find the actual solution to linear problems.
//!The number of solvers available in this module depends on which cargo features you have activated.

#[cfg(feature = "coin_cbc")]
#[cfg_attr(docsrs, doc(cfg(feature = "coin_cbc")))]
pub mod coin_cbc;

#[cfg(feature = "minilp")]
#[cfg_attr(docsrs, doc(cfg(feature = "minilp")))]
pub mod minilp;

#[cfg(feature = "lpsolve")]
#[cfg_attr(docsrs, doc(cfg(feature = "lpsolve")))]
pub mod lpsolve;

#[cfg(feature = "highs")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub mod highs;

use crate::{constraint::ConstraintReference, Variable};
use crate::{Constraint, Expression};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Whether to search for the variable values that give the highest
/// or the lowest value of the objective function.
#[derive(Eq, PartialEq, Clone, Copy)]
pub enum ObjectiveDirection {
    /// Find the highest possible value of the objective
    Maximisation,
    /// Find the lowest possible value of the objective
    Minimisation,
}

/// Represents an error that occurred when solving a problem.
///
/// # Examples
/// ## Infeasible
/// ```
/// use good_lp::*;
/// let mut vars = variables!();
/// let x = vars.add_variable(); // unbounded variable
/// let result = vars.maximise(x)
///              .using(default_solver)
///              .with(constraint!(x <= 9))
///              .with(constraint!(x >= 10))
///              .solve(); // x cannot be less than 9 and more than 10 at the same time
/// assert_eq!(result.err(), Some(ResolutionError::Infeasible));
/// ```
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

impl Display for ResolutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionError::Unbounded =>
                write!(f, "Unbounded: The objective can be made infinitely large without violating any constraints."),
            ResolutionError::Infeasible =>
                write!(f, "Infeasible: The problem contains contradictory constraints. No solution exists."),
            ResolutionError::Other(s) =>
                write!(f, "An unexpected error occurred while running the optimizer: {}.", s)
        }
    }
}

impl Error for ResolutionError {}

/// A solver's own representation of a model, to which constraints can be added.
pub trait SolverModel {
    /// The type of the solution to the problem
    type Solution: Solution;
    /// The error that can occur while solving the problem
    type Error;

    /// Takes a model and adds a constraint to it
    fn with(mut self, constraint: Constraint) -> Self
    where
        Self: Sized,
    {
        self.add_constraint(constraint);
        self
    }

    /// Find the solution for the problem being modeled
    fn solve(self) -> Result<Self::Solution, Self::Error>;

    /// Adds a constraint to the Model and returns a reference to the index
    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference;
}

/// A problem solution
pub trait Solution {
    /// Get the optimal value of a variable of the problem
    fn value(&self, variable: Variable) -> f64;

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
    fn eval(&self, expr: &Expression) -> f64
    where
        Self: Sized,
    {
        expr.eval_with(self)
    }
}

/// All `HashMap<Variable, {number}>` implement [Solution].
/// If a HashMap doesn't contain the value for a variable,
/// then [Solution::value] will panic if you try to access it.
impl<N: Into<f64> + Clone> Solution for HashMap<Variable, N> {
    fn value(&self, variable: Variable) -> f64 {
        self[&variable].clone().into()
    }
}

/// A Solution that supports Dual values
pub trait SolutionWithDual {
    /// Method to retrieve a single Dual Value for a given constraint
    fn dual(&self, c: ConstraintReference) -> f64;
}
