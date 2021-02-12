use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use crate::expression::{Expression, LinearExpression};
use crate::solvers::ObjectiveDirection;

/// A variable in a problem
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

pub trait FormatWithVars<F> {
    fn format_with<FUN>(
        &self,
        f: &mut Formatter<'_>,
        variable_format: FUN,
    ) -> std::fmt::Result
        where FUN: Fn(&mut Formatter<'_>, Variable<F>) -> std::fmt::Result;

    fn format_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_with(f, |f, var| {
            write!(f, "v{}", var.index())
        })
    }
}

pub struct VariableDefinition;

/// Represents the variable for a given problem.
/// Each problem has a unique type, which prevents using the variables
/// from one problem inside an other one
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

    pub fn add_variable(&mut self) -> Variable<F> {
        let index = self.variables.len();
        self.variables.push(VariableDefinition);
        Variable { _problem_type: PhantomData, index }
    }

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

    pub fn maximise<E: Into<Expression<F>>>(self, objective: E) -> UnsolvedProblem<F> {
        self.optimise(ObjectiveDirection::Maximisation, objective)
    }

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

#[macro_export]
macro_rules! variables {
    () => {
        $crate::variable::ProblemVariables::__new_internal(||())
    }
}

pub struct UnsolvedProblem<F> {
    pub(crate) objective: Expression<F>,
    pub(crate) direction: ObjectiveDirection,
    pub(crate) variables: ProblemVariables<F>,
}

impl<F> UnsolvedProblem<F> {
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

impl<F> Add<Variable<F>> for f64 {
    type Output = Expression<F>;
    fn add(self, rhs: Variable<F>) -> Self::Output { rhs + self }
}

impl<F> Add<Variable<F>> for i32 {
    type Output = Expression<F>;
    fn add(self, rhs: Variable<F>) -> Self::Output { rhs + self }
}

impl<F> Div<f64> for Variable<F> {
    type Output = Expression<F>;
    fn div(self, rhs: f64) -> Self::Output { self * (1. / rhs) }
}

impl<F> Div<i32> for Variable<F> {
    type Output = Expression<F>;
    fn div(self, rhs: i32) -> Self::Output { self * (1. / f64::from(rhs)) }
}

impl<F, N: Into<f64>> Add<N> for Variable<F> {
    type Output = Expression<F>;

    fn add(self, rhs: N) -> Self::Output {
        let mut expr = Expression::from(self);
        expr += rhs;
        expr
    }
}

impl<F> Add<Variable<F>> for Variable<F> {
    type Output = Expression<F>;

    fn add(self, rhs: Variable<F>) -> Self::Output {
        if rhs == self {
            self * 2
        } else {
            Expression::from(self) + rhs
        }
    }
}

impl<F> Add<Expression<F>> for Variable<F> {
    type Output = Expression<F>;

    fn add(self, rhs: Expression<F>) -> Self::Output {
        rhs + Expression::from(self)
    }
}

impl<F> Sub<Expression<F>> for Variable<F> {
    type Output = Expression<F>;

    fn sub(self, rhs: Expression<F>) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl<F> Sub<Variable<F>> for Variable<F> {
    type Output = Expression<F>;

    fn sub(self, rhs: Variable<F>) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl<F> Sub<Variable<F>> for f64 {
    type Output = Expression<F>;

    fn sub(self, rhs: Variable<F>) -> Self::Output {
        self - Expression::from(rhs)
    }
}
