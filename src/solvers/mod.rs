//! Included solvers that find the actual solution to linear problems.
//!The number of solvers available in this module depends on which cargo features you have activated.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use crate::variable::UnsolvedProblem;
use crate::Constraint;
use crate::{constraint::ConstraintReference, IntoAffineExpression, Variable};

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

/// An entity that is able to solve linear problems
pub trait Solver {
    /// The internal model type used by the solver
    type Model: SolverModel;
    /// Solve the given problem
    fn create_model(&mut self, problem: UnsolvedProblem) -> Self::Model;
}

/// A solver that is valid for the static lifetime
pub trait StaticSolver: Solver + 'static {}

impl<T> StaticSolver for T where T: Solver + 'static {}

/// A function that takes an [UnsolvedProblem] and returns a [SolverModel] automatically implements [Solver]
impl<SOLVER, MODEL> Solver for SOLVER
where
    SOLVER: FnMut(UnsolvedProblem) -> MODEL,
    MODEL: SolverModel,
{
    type Model = MODEL;
    fn create_model(&mut self, pb: UnsolvedProblem) -> Self::Model {
        self(pb)
    }
}

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
    type Error: std::error::Error;

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
    fn eval<E: IntoAffineExpression>(&self, expr: E) -> f64
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

/// A type that contains the dual values of a solution.
/// See [SolutionWithDual].
pub trait DualValues {
    /// Retrieve a single dual value for a given constraint.
    /// This returns the value of the solution for the corresponding variable in the dual problem.
    /// This is also called "shadow price" or "dual price".
    fn dual(&self, c: ConstraintReference) -> f64;
}

/// The dual value measures the increase in the objective function's value per unit
/// increase in the variable's value. The dual value for a constraint is nonzero only when
/// the constraint is equal to its bound. Also known as the shadow price.
/// This trait handles the retrieval of dual values from a solver.
pub trait SolutionWithDual<'a> {
    /// Type of the object containing the dual values.
    type Dual: DualValues;
    /// Get the dual values for a problem.
    /// If a solver requires running additional computations or allocating additional memory
    /// to get the dual values, this is performed when running this method.
    fn compute_dual(&'a mut self) -> Self::Dual;
}

/// A model that supports [SOS type 1](https://en.wikipedia.org/wiki/Special_ordered_set) constraints.
#[allow(clippy::upper_case_acronyms)]
pub trait ModelWithSOS1 {
    /// Adds a constraint saying that two variables from the given set cannot be non-zero at once.
    ///
    /// ```
    /// use good_lp::*;
    /// # // Not all solvers support SOS constraints
    /// # #[cfg(any(feature = "lpsolve", feature = "coin_cbc"))] {
    /// # let solver = default_solver;
    /// variables! {problem:
    ///     0 <= x <= 2;
    ///     0 <= y <= 3;
    /// }
    /// let solution = problem
    ///     .maximise(x + y) // maximise x + y
    ///     .using(solver)
    ///     .with_sos1(x + y) // but require that either x or y is zero
    ///     .solve().unwrap();
    /// assert_eq!(solution.value(x), 0.);
    /// assert_eq!(solution.value(y), 3.);
    /// # }
    /// ```
    fn add_sos1<I: IntoAffineExpression>(&mut self, variables_and_weights: I);

    /// See [ModelWithSOS1::add_sos1]
    fn with_sos1<I: IntoAffineExpression>(mut self, variables_and_weights: I) -> Self
    where
        Self: Sized,
    {
        self.add_sos1(variables_and_weights);
        self
    }
}
