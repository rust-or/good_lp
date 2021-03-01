//! A [Variable] is the base element used to create an [Expression].
//! The goal of the solver is to find optimal values for all variables in a problem.
//!
//! Each variable has a [VariableDefinition] that sets its bounds.
use std::collections::{Bound, HashMap};
use std::fmt::{Debug, Formatter};
use std::hash::Hash;
use std::ops::{Div, Mul, Neg, RangeBounds};

use crate::affine_expression_trait::IntoAffineExpression;
use crate::expression::{Expression, LinearExpression};
use crate::solvers::ObjectiveDirection;

/// A variable in a problem. Use variables to create [expressions](Expression),
/// to express the [objective](ProblemVariables::optimise)
/// and the [Constraints](crate::Constraint) of your model.
///
/// Variables are created using [ProblemVariables::add]
///
/// ## Warning
/// `Eq` is implemented on this type, but
/// `v1 == v2` is true only if the two variables represent the same object,
/// not if they have the same definition.
///
/// ```
/// # use good_lp::{variable, variables};
/// let mut vars = variables!();
/// let v1 = vars.add(variable().min(1).max(8));
/// let v2 = vars.add(variable().min(1).max(8));
/// assert_ne!(v1, v2);
///
/// let v1_copy = v1;
/// assert_eq!(v1, v1_copy);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable {
    /// A variable is nothing more than an index into the `variables` field of a ProblemVariables
    /// That's why it can be `Copy`.
    /// All the actual information about the variable (name, type, bounds, ...) is stored in ProblemVariables
    index: usize,
}

impl IntoAffineExpression for Variable {
    type Iter = std::iter::Once<(Self, f64)>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        std::iter::once((self, 1.))
    }
}

impl<'a> IntoAffineExpression for &'a Variable {
    type Iter = std::iter::Once<(Variable, f64)>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        (*self).linear_coefficients()
    }
}

impl Variable {
    /// No one should use this method outside of [VariableDefinition]
    fn at(index: usize) -> Self {
        Self { index }
    }
}

impl Variable {
    pub(super) fn index(&self) -> usize {
        self.index
    }
}

/// An element that can be displayed if you give a variable display function
pub trait FormatWithVars {
    /// Write the element to the formatter. See [std::fmt::Display]
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: Fn(&mut Formatter<'_>, Variable) -> std::fmt::Result;

    /// Write the elements, naming the variables v0, v1, ... vn
    fn format_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_with(f, |f, var| write!(f, "v{}", var.index()))
    }
}

/// Defines the properties of a variable, such as its lower and upper bounds.
#[derive(Clone, PartialEq, Debug)]
pub struct VariableDefinition {
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl VariableDefinition {
    /// Creates an unbounded continuous linear variable
    pub fn new() -> Self {
        VariableDefinition {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
    }

    /// Set the lower and/or higher bounds of the variable
    ///
    /// ## Examples
    /// ```
    /// # use good_lp::variable;
    /// assert_eq!(
    ///     variable().bounds(1..2),
    ///     variable().min(1).max(2)
    /// );
    ///
    /// assert_eq!(
    ///     variable().bounds(1..),
    ///     variable().min(1)
    /// );
    ///
    /// assert_eq!(
    ///     variable().bounds(..=2),
    ///     variable().max(2)
    /// );
    ///
    /// # assert_eq!(variable().bounds::<f64, _>(..), variable());
    /// ```
    pub fn bounds<N: Into<f64> + Copy, B: RangeBounds<N>>(self, bounds: B) -> Self {
        self.min(match bounds.start_bound() {
            Bound::Included(&x) => x.into(),
            Bound::Excluded(&x) => x.into(),
            Bound::Unbounded => f64::NEG_INFINITY,
        })
        .max(match bounds.end_bound() {
            Bound::Included(&x) => x.into(),
            Bound::Excluded(&x) => x.into(),
            Bound::Unbounded => f64::INFINITY,
        })
    }

    /// Set the lower bound of the variable
    pub fn min<N: Into<f64>>(mut self, min: N) -> Self {
        self.min = min.into();
        self
    }
    /// Set the higher bound of the variable
    pub fn max<N: Into<f64>>(mut self, max: N) -> Self {
        self.max = max.into();
        self
    }

    /// Set both the lower and higher bounds of the variable
    pub fn clamp<N1: Into<f64>, N2: Into<f64>>(self, min: N1, max: N2) -> Self {
        self.min(min).max(max)
    }
}

/// Creates an unbounded continuous linear variable
impl Default for VariableDefinition {
    fn default() -> Self {
        VariableDefinition::new()
    }
}

/// Returns an anonymous unbounded continuous variable definition
pub fn variable() -> VariableDefinition {
    VariableDefinition::default()
}

/// Represents the variables for a given problem.
/// Each problem has a unique type, which prevents using the variables
/// from one problem inside an other one.
/// Instances of this type should be created exclusively using the [variables!] macro.
#[derive(Default)]
pub struct ProblemVariables {
    variables: Vec<VariableDefinition>,
}

impl ProblemVariables {
    /// Create an empty list of variables
    pub fn new() -> Self {
        ProblemVariables { variables: vec![] }
    }

    /// Add a anonymous unbounded continuous variable to the problem
    pub fn add_variable(&mut self) -> Variable {
        self.add(variable())
    }

    /// Add a variable with the given definition
    ///
    /// ```
    /// # use good_lp::*;
    /// variables!{problem: 2 <= x <= 3;}
    /// ```
    /// is equivalent to
    /// ```
    /// # use good_lp::*;
    /// let mut problem = ProblemVariables::new();
    /// let y = problem.add(variable().min(0));
    /// ```
    pub fn add(&mut self, var_def: VariableDefinition) -> Variable {
        let index = self.variables.len();
        self.variables.push(var_def);
        Variable::at(index)
    }

    /// Adds a list of variables with the given definition
    ///
    /// ```
    /// use good_lp::*;
    /// // Solve a problem with 11 variables: x, y0, y1, ..., y9
    /// variables!{problem: 2 <= x <= 3;}
    /// let y: Vec<Variable> = problem.add_vector(variable().min(0), 10);
    /// let objective: Expression = y.iter().sum(); // Minimise sum(y_i for i in [0; 9])
    /// let mut model = problem.minimise(objective).using(default_solver);
    /// // for all i, we must have y_i >= x
    /// for y_i in y.iter() {
    ///   model = model.with(constraint!(y_i >= x));
    /// }
    /// let solution = model.solve().unwrap();
    /// assert_eq!(solution.value(y[3]), 2.);
    /// ```
    pub fn add_vector(&mut self, var_def: VariableDefinition, len: usize) -> Vec<Variable> {
        (0..len).map(|_i| self.add(var_def.clone())).collect()
    }

    /// Creates an optimization problem with the given objective. Don't solve it immediately.
    ///
    /// ```
    /// use good_lp::{variables, variable, default_solver, SolverModel, Solution};
    /// use good_lp::solvers::ObjectiveDirection;
    /// fn solve(sense: ObjectiveDirection) -> f64 {
    ///    variables!{problem: 2 <= x <= 3;}
    ///     let solution = problem.optimise(sense, x).using(default_solver).solve().unwrap();
    ///     solution.value(x)
    /// }
    /// assert_eq!(solve(ObjectiveDirection::Minimisation), 2.);
    /// assert_eq!(solve(ObjectiveDirection::Maximisation), 3.);
    /// ```
    pub fn optimise<E: IntoAffineExpression>(
        self,
        direction: ObjectiveDirection,
        objective: E,
    ) -> UnsolvedProblem {
        let objective = Expression::from_other_affine(objective);
        assert!(
            objective.linear.coefficients.len() <= self.variables.len(),
            "There should not be more variables in the objective function than in the problem. \
            You probably used variables from a different problem in this one."
        );
        UnsolvedProblem {
            objective,
            direction,
            variables: self,
        }
    }

    /// Creates an maximization problem with the given objective. Don't solve it immediately
    ///
    /// ```
    /// use good_lp::{variables, variable, default_solver, SolverModel, Solution};
    /// variables!{problem: x <= 7;}
    /// let solution = problem.maximise(x).using(default_solver).solve().unwrap();
    /// assert_eq!(solution.value(x), 7.);
    /// ```
    pub fn maximise<E: IntoAffineExpression>(self, objective: E) -> UnsolvedProblem {
        self.optimise(ObjectiveDirection::Maximisation, objective)
    }

    /// Creates an minimization problem with the given objective. Don't solve it immediately
    /// ```
    /// use good_lp::{variables, variable, default_solver, SolverModel, Solution};
    /// variables!{problem: x >= -8;}
    /// let solution = problem.minimise(x).using(default_solver).solve().unwrap();
    /// assert_eq!(solution.value(x), -8.);
    /// ```
    pub fn minimise<E: IntoAffineExpression>(self, objective: E) -> UnsolvedProblem {
        self.optimise(ObjectiveDirection::Minimisation, objective)
    }

    /// Iterates over the couples of variables with their properties
    pub fn iter_variables_with_def(&self) -> impl Iterator<Item = (Variable, &VariableDefinition)> {
        self.variables
            .iter()
            .enumerate()
            .map(|(i, def)| (Variable::at(i), def))
    }

    /// The number of variables
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Returns true when no variables have been added
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }
}

impl IntoIterator for ProblemVariables {
    type Item = VariableDefinition;
    type IntoIter = std::vec::IntoIter<VariableDefinition>;

    fn into_iter(self) -> Self::IntoIter {
        self.variables.into_iter()
    }
}

/// A problem without constraints.
/// Created with [ProblemVariables::optimise].
pub struct UnsolvedProblem {
    pub(crate) objective: Expression,
    pub(crate) direction: ObjectiveDirection,
    pub(crate) variables: ProblemVariables,
}

impl UnsolvedProblem {
    /// Create a solver instance and feed it with this problem
    pub fn using<S, G>(self, solver: S) -> G
    where
        S: FnOnce(UnsolvedProblem) -> G,
    {
        solver(self)
    }
}

impl<N: Into<f64>> Mul<N> for Variable {
    type Output = Expression;

    fn mul(self, rhs: N) -> Self::Output {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(self, rhs.into());
        Expression {
            linear: LinearExpression { coefficients },
            constant: 0.0,
        }
    }
}

impl Mul<Variable> for f64 {
    type Output = Expression;

    fn mul(self, rhs: Variable) -> Self::Output {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(rhs, self);
        Expression {
            linear: LinearExpression { coefficients },
            constant: 0.0,
        }
    }
}

impl Mul<Variable> for i32 {
    type Output = Expression;

    fn mul(self, rhs: Variable) -> Self::Output {
        rhs.mul(f64::from(self))
    }
}

impl Div<f64> for Variable {
    type Output = Expression;
    fn div(self, rhs: f64) -> Self::Output {
        self * (1. / rhs)
    }
}

impl Div<i32> for Variable {
    type Output = Expression;
    fn div(self, rhs: i32) -> Self::Output {
        self * (1. / f64::from(rhs))
    }
}

impl Neg for Variable {
    type Output = Expression;

    fn neg(self) -> Self::Output {
        -Expression::from(self)
    }
}
