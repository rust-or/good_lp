//! A [Variable] is the base element used to create an [Expression].
//! The goal of the solver is to find optimal values for all variables in a problem.
//!
//! Each variable has a [VariableDefinition] that sets its bounds.
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Div, Mul, Neg};

use crate::expression::{Expression, LinearExpression};
use crate::solvers::ObjectiveDirection;

/// A variable in a problem. Use variables to create [expressions](Expression),
/// to express the [objective](ProblemVariables::optimise) and the [Constraints](crate::Constraint) of your model.
#[derive(Debug, Default)]
pub struct Variable<T> {
    _problem_type: PhantomData<T>,
    /// A variable is nothing more than an index into the `variables` field of a ProblemVariables
    /// That's why it can be `Copy`.
    /// All the actual information about the variable (name, type, bounds, ...) is stored in ProblemVariables
    index: usize,
}

impl<T> Variable<T> {
    pub(super) fn index(&self) -> usize { self.index }
}

/// This checks if two variables are the same (or copies one of another)
/// This is **not** a check that the two variables have the same [VariableDefinition]
impl<F> PartialEq for Variable<F> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}


impl<F> Eq for Variable<F> {}

impl<F> Hash for Variable<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state)
    }
}


impl<F> Clone for Variable<F> {
    fn clone(&self) -> Self {
        Self { _problem_type: PhantomData, index: self.index }
    }
}

impl<F> Copy for Variable<F> {}

/// An element that can be displayed if you give a variable display function
pub trait FormatWithVars<F> {
    /// Write the element to the formatter. See [std::fmt::Display]
    fn format_with<FUN>(
        &self,
        f: &mut Formatter<'_>,
        variable_format: FUN,
    ) -> std::fmt::Result
        where FUN: Fn(&mut Formatter<'_>, Variable<F>) -> std::fmt::Result;

    /// Write the elements, naming the variables v0, v1, ... vn
    fn format_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_with(f, |f, var| {
            write!(f, "v{}", var.index())
        })
    }
}

/// Defines the properties of a variable, such as its lower and upper bounds.
#[derive(Clone, PartialEq)]
pub struct VariableDefinition {
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl VariableDefinition {
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
}

/// Creates an unbounded continuous linear variable
impl Default for VariableDefinition {
    fn default() -> Self {
        VariableDefinition {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
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
pub struct ProblemVariables<F> {
    _type_signature: F,
    variables: Vec<VariableDefinition>,
}

impl<F: Fn()> ProblemVariables<F> {
    /// This method has to be exposed for the variables! macro to work,
    /// but it should **never** be called directly
    #[doc(hidden)]
    pub fn __new_internal(_type_signature: F) -> Self {
        ProblemVariables { _type_signature, variables: vec![] }
    }

    /// Add a anonymous unbounded continuous variable to the problem
    pub fn add_variable(&mut self) -> Variable<F> {
        self.add(variable())
    }

    /// Add a variable with the given definition
    pub fn add(&mut self, var_def: VariableDefinition) -> Variable<F> {
        let index = self.variables.len();
        self.variables.push(var_def);
        Variable { _problem_type: PhantomData, index }
    }

    /// Adds a list of variables with the given definition
    pub fn add_vector(&mut self, var_def: VariableDefinition, len: usize) -> Vec<Variable<F>> {
        (0..len).map(|_i| {
            self.add(var_def.clone())
        }).collect()
    }

    /// Creates an optimization problem with the given objective. Don't solve it immediately
    pub fn optimise<E: Into<Expression<F>>>(
        self,
        direction: ObjectiveDirection,
        objective: E,
    ) -> UnsolvedProblem<F> {
        UnsolvedProblem {
            objective: objective.into(),
            direction,
            variables: self,
        }
    }

    /// Creates an maximization problem with the given objective. Don't solve it immediately
    pub fn maximise<E: Into<Expression<F>>>(self, objective: E) -> UnsolvedProblem<F> {
        self.optimise(ObjectiveDirection::Maximisation, objective)
    }

    /// Creates an minimization problem with the given objective. Don't solve it immediately
    pub fn minimise<E: Into<Expression<F>>>(self, objective: E) -> UnsolvedProblem<F> {
        self.optimise(ObjectiveDirection::Minimisation, objective)
    }
}

impl<F> IntoIterator for ProblemVariables<F>{
    type Item = VariableDefinition;
    type IntoIter = std::vec::IntoIter<VariableDefinition>;

    fn into_iter(self) -> Self::IntoIter {
        self.variables.into_iter()
    }
}

/// A problem without constraints
pub struct UnsolvedProblem<F> {
    pub(crate) objective: Expression<F>,
    pub(crate) direction: ObjectiveDirection,
    pub(crate) variables: ProblemVariables<F>,
}

impl<F> UnsolvedProblem<F> {
    /// Create a solver instance and feed it with this problem
    pub fn using<S, G>(self, solver: S) -> G
        where S: FnOnce(UnsolvedProblem<F>) -> G {
        solver(self)
    }
}

impl<F, N: Into<f64>> Mul<N> for Variable<F> {
    type Output = Expression<F>;

    fn mul(self, rhs: N) -> Self::Output {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(self, rhs.into());
        Expression { linear: LinearExpression { coefficients }, constant: 0.0 }
    }
}


impl<F> Mul<Variable<F>> for f64 {
    type Output = Expression<F>;

    fn mul(self, rhs: Variable<F>) -> Self::Output {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(rhs, self);
        Expression { linear: LinearExpression { coefficients }, constant: 0.0 }
    }
}

impl<F> Mul<Variable<F>> for i32 {
    type Output = Expression<F>;

    fn mul(self, rhs: Variable<F>) -> Self::Output {
        rhs.mul(f64::from(self))
    }
}

impl<F> Div<f64> for Variable<F> {
    type Output = Expression<F>;
    fn div(self, rhs: f64) -> Self::Output { self * (1. / rhs) }
}

impl<F> Div<i32> for Variable<F> {
    type Output = Expression<F>;
    fn div(self, rhs: i32) -> Self::Output { self * (1. / f64::from(rhs)) }
}

impl<T> Neg for Variable<T> {
    type Output = Expression<T>;

    fn neg(self) -> Self::Output {
        -Expression::from(self)
    }
}