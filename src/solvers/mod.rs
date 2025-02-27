//! Included solvers that find the actual solution to linear problems.
//!The number of solvers available in this module depends on which cargo features you have activated.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};

use crate::variable::UnsolvedProblem;
use crate::Constraint;
use crate::{constraint::ConstraintReference, IntoAffineExpression, Variable};

#[cfg(feature = "cplex-rs")]
#[cfg_attr(docsrs, doc(cfg(feature = "cplex-rs")))]
pub mod cplex;

#[cfg(feature = "coin_cbc")]
#[cfg_attr(docsrs, doc(cfg(feature = "coin_cbc")))]
pub mod coin_cbc;

#[cfg(feature = "microlp")]
#[cfg_attr(docsrs, doc(cfg(feature = "microlp")))]
pub mod microlp;

#[cfg(feature = "lpsolve")]
#[cfg_attr(docsrs, doc(cfg(feature = "lpsolve")))]
pub mod lpsolve;

#[cfg(feature = "highs")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub mod highs;

#[cfg(feature = "scip")]
#[cfg_attr(docsrs, doc(cfg(feature = "scip")))]
pub mod scip;

#[cfg(feature = "lp-solvers")]
#[cfg_attr(docsrs, doc(cfg(feature = "lp-solvers")))]
pub mod lp_solvers;

#[cfg(feature = "clarabel")]
#[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
pub mod clarabel;

/// An entity that is able to solve linear problems
pub trait Solver {
    /// The internal model type used by the solver
    type Model: SolverModel;
    /// Solve the given problem
    fn create_model(&mut self, problem: UnsolvedProblem) -> Self::Model;

    /// The human readable name of the solver, for instance "Coin Cbc"
    fn name() -> &'static str;
}

/// Returns the name of a solver
///
/// ```
/// # #[cfg(feature = "coin_cbc")] {
/// use good_lp::*;
/// assert_eq!(solver_name(default_solver), "Coin Cbc");
/// }
/// ```
pub fn solver_name<T: Solver>(_: T) -> &'static str {
    <T as Solver>::name()
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

    fn name() -> &'static str {
        MODEL::name()
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
    /// An error string
    Str(String),
}

impl Display for ResolutionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolutionError::Unbounded =>
                write!(f, "Unbounded: The objective can be made infinitely large without violating any constraints."),
            ResolutionError::Infeasible =>
                write!(f, "Infeasible: The problem contains contradictory constraints. No solution exists."),
            ResolutionError::Other(s) =>
                write!(f, "An unexpected error occurred while running the optimizer: {}.", s),
            ResolutionError::Str(s) =>
                write!(f, "An unexpected error occurred while running the optimizer: {}.", s)
        }
    }
}

impl From<String> for ResolutionError {
    fn from(s: String) -> Self {
        ResolutionError::Str(s)
    }
}

impl Error for ResolutionError {}

/// Represents an error setting the MIP gap

#[derive(Debug, PartialEq, Clone)]
pub enum MipGapError {
    /// The MIP gap is negative (must be >= 0)
    Negative,
    /// The MIP gap is infinite (must be finite)
    Infinite,
    /// Another error occurred
    Other(String),
}

impl Display for MipGapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MipGapError::Negative => write!(f, "Negative: The MIP gap is negative"),
            MipGapError::Infinite => write!(f, "Infinite: The MIP gap is infinite"),
            MipGapError::Other(s) => {
                write!(f, "An unexpected error occurred setting the MIP Gap: {s}")
            }
        }
    }
}

impl Error for MipGapError {}

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

    /// Takes a model and adds a list of constraints to it
    ///
    /// # Examples
    /// ```rust
    /// use good_lp::*;
    /// let mut vars = variables!();
    /// let x = vars.add_variable(); // unbounded variable
    /// let epsilon = 1e-7; // works if epsilon is no smaller than this
    /// let result = vars.maximise(x)
    ///              .using(default_solver)
    ///              .with_all([constraint!(x >= 1.), constraint!(x <= 10.)])
    ///              .solve()
    ///              .expect("example model, trivial to solve"); //
    /// assert!((result.eval(&x) - 10.).abs() <= epsilon)
    /// ```
    fn with_all(mut self, constraints: impl IntoIterator<Item = Constraint>) -> Self
    where
        Self: Sized,
    {
        for constraint in constraints {
            self.add_constraint(constraint);
        }
        self
    }

    /// Find the solution for the problem being modeled
    fn solve(self) -> Result<Self::Solution, Self::Error>;

    /// Adds a constraint to the Model and returns a reference to the index
    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference;

    /// Human readable name of the solver, for instance "Coin Cbc"
    fn name() -> &'static str;
}

/// A solver that can take an initial solution to a problem before solving it
pub trait WithInitialSolution {
    /// Sets the initial solution to the problem
    fn with_initial_solution(self, solution: impl IntoIterator<Item = (Variable, f64)>) -> Self;
}

/// A solver than can stop the solving process after some time
pub trait WithTimeLimit {
    /// Sets the time limit for the solver
    fn with_time_limit<T: Into<f64>>(self, seconds: T) -> Self;
}

/// Information about the status of a solution, such as whether the solution is
/// optimal
#[derive(Clone, Copy, Debug)]
pub enum SolutionStatus {
    /// The solution is optimal
    Optimal,
    /// The solution is not optimal and it was obtained because the time limit
    /// was reached
    TimeLimit,
}

/// A problem solution
pub trait Solution {
    /// Returns `true` if this solution is optimal and `false` otherwise
    fn status(&self) -> SolutionStatus;

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
    fn status(&self) -> SolutionStatus {
        SolutionStatus::Optimal
    }
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
/// increase in a constraint's value.
/// The dual value for a constraint is nonzero only when
/// the constraint is equal to its bound. Also known as the shadow price.
///
/// It is useful for understanding "how limiting" a constraint is.
///
/// This trait handles the retrieval of dual values from a solver.
///
/// ## Example
///
/// ```
/// use good_lp::*;
/// # // These solvers do not support dual values
/// # #[cfg(not(any(
/// #     feature = "coin_cbc",
/// #     feature = "microlp",
/// #     feature = "lpsolve",
/// #     feature = "lp-solvers",
/// #     feature = "scip",
/// #     feature = "cplex-rs",
/// # )))] {
///
/// variables!{
///    vars:
///     0 <= a <= 1;
///     0 <= b <= 4;
/// };
/// let mut pb = vars.maximise(a + b).using(default_solver);
/// let c1 = pb.add_constraint(constraint!(a + 2*b <= 5));
/// let non_binding = pb.add_constraint(constraint!(a + b <= 30));
/// let mut solution = pb.solve().unwrap();
/// let dual = solution.compute_dual();
/// # use float_eq::assert_float_eq;
/// // The dual value of c1 is 0.5, because the constraint is binding, and the objective function
/// // increases by 0.5 for each unit increase in the constraint.
/// // I.e. a+b is currently maximised at 3 for a=1, b=2.
/// // If we increase the constraint by 1, setting a+2*b<=6, we would have a=1, b=2.5, and a+b=3.5.
/// // The increase in the objective function is 3.5-3=0.5, for a 1 unit increase in the constraint.
/// assert_float_eq!(dual.dual(c1), 0.5, abs <= 1e-8);
///
/// // The dual value of non_binding is 0, because the constraint is not binding.
/// // The objective function does not change if we increase the constraint.
/// assert_float_eq!(dual.dual(non_binding), 0., abs <= 1e-8);
/// # }
/// ```
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

/// A model that supports setting the MIP gap
///
/// Setting the MIP gap can cause the solver to return a solution faster at the
/// expense of being suboptimal within a specified tolerance.  Solvers vary in
/// their definition of the relative MIP gap but common definitions are
///
/// |UpperBound - LowerBound| / |UpperBound| *or* |UpperBound - LowerBound| / |LowerBound|
///
/// where, for maximisation, UpperBound is the upper bound of the relaxed solution
/// and LowerBound is the lower bound of the integer solution.
///
/// For example, setting the MIP gap to 0.1 would return a solution that's within
/// 10% of the solver's estimate of the best possible solution.
pub trait WithMipGap {
    /// Get the relative MIP gap
    fn mip_gap(&self) -> Option<f32>;

    /// Set the relative MIP gap
    ///
    /// ```
    /// // Knapsack problem
    /// //
    /// // Given a set of objects, each with a value and a cost, find the subset of
    /// // objects that maximises total value without exceeding a total cost budget
    ///
    /// use good_lp::*;
    /// # // Not all solvers support setting the MIP gap
    /// # #[cfg(any(feature = "highs", feature = "coin_cbc"))] {
    /// # let solver = default_solver;
    ///
    /// // (value, cost) of each object
    /// let objects: Vec<(f64, f64)> = vec![
    ///     (1.87, 6.03),
    ///     (3.22, 8.03),
    ///     (9.91, 5.16),
    ///     (8.31, 1.72),
    ///     (7.00, 6.33),
    ///     (5.15, 8.20),
    ///     (8.01, 4.63),
    ///     (2.22, 1.50),
    ///     (7.04, 6.26),
    ///     (8.99, 9.62),
    ///     (2.13, 4.00),
    ///     (8.02, 8.02),
    ///     (3.07, 1.92),
    ///     (1.98, 9.03),
    ///     (7.23, 9.51),
    ///     (4.08, 3.24),
    ///     (9.65, 5.13),
    ///     (6.53, 3.07),
    ///     (6.76, 3.84),
    ///     (9.63, 8.33),
    /// ];
    ///
    /// let budget: f64 = 25.0;
    ///
    /// let value_optimal = knapsack_value(solver, &objects, budget, None);
    /// let value_suboptimal = knapsack_value(solver, &objects, budget, Some(0.5));
    ///
    /// // NOTE: this assertion may fail if the solver finds an optimal solution
    /// // before it checks the MIP gap
    /// assert!(value_suboptimal < value_optimal);
    ///
    /// fn knapsack_value<S>(
    ///     solver: S,
    ///     objects: &[(f64, f64)],
    ///     budget: f64,
    ///     mipgap: Option<f32>,
    /// ) -> f64
    /// where
    ///     S: Solver,
    ///     S::Model: SolverModel + WithMipGap,
    /// {
    ///     let mut prob_vars = ProblemVariables::new();
    ///     let mut objective = Expression::with_capacity(objects.len());
    ///     let mut constraint = Expression::with_capacity(objects.len());
    ///
    ///     for (value, cost) in objects {
    ///         let var = prob_vars.add(variable().binary());
    ///         objective.add_mul(*value, var);
    ///         constraint.add_mul(*cost, var);
    ///     }
    ///
    ///     let mut model = prob_vars.maximise(objective.clone()).using(solver);
    ///
    ///     if let Some(gap) = mipgap {
    ///         model = model.with_mip_gap(gap).unwrap();
    ///     }
    ///
    ///     model.add_constraint(constraint.leq(budget));
    ///
    ///     let solution = model.solve().unwrap();
    ///
    ///     // For this example we're interested only in the total value, not in the objects selected
    ///     objective.eval_with(&solution)
    /// }
    /// # }
    /// ```
    fn with_mip_gap(self, mip_gap: f32) -> Result<Self, MipGapError>
    where
        Self: Sized;
}
