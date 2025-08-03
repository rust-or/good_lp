//! A [Variable] is the base element used to create an [Expression].
//! The goal of the solver is to find optimal values for all variables in a problem.
//!
//! Each variable has a [VariableDefinition] that sets its bounds.
use std::collections::Bound;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::iter::{repeat, FromIterator};
use std::ops::{Div, Mul, Neg, Not, RangeBounds};

use fnv::FnvHashMap as HashMap;

use crate::affine_expression_trait::IntoAffineExpression;
use crate::expression::{Expression, LinearExpression};
use crate::solvers::{ObjectiveDirection, Solver};

/// A variable in a problem. Use variables to create [expressions](Expression),
/// to express the [objective](ProblemVariables::optimise)
/// and the [Constraints](crate::Constraint) of your model.
///
/// Variables are created using [ProblemVariables::add]. They implement std::ops basic math operations on f64 and i32 values.
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

/// Use an optional variable as an expression
///
/// ```
/// # use good_lp::variables;
/// variables! {problem: 0 <= v};
/// let maybe = Some(v);
/// problem.minimise(v + maybe);
/// ```
impl IntoAffineExpression for Option<Variable> {
    #[allow(clippy::type_complexity)]
    type Iter = std::iter::Map<std::option::IntoIter<Variable>, fn(Variable) -> (Variable, f64)>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.into_iter().map(|v| (v, 1.))
    }
}

impl IntoAffineExpression for &Variable {
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
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result;

    /// Write the elements, naming the variables v0, v1, ... vn
    fn format_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_with(f, |f, var| write!(f, "v{}", var.index()))
    }
}

impl FormatWithVars for Variable {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        variable_format(f, *self)
    }
}

/// Defines the properties of a variable, such as its lower and upper bounds.
#[derive(Clone, PartialEq, Debug)]
pub struct VariableDefinition {
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) initial: Option<f64>,
    pub(crate) name: String,
    pub(crate) is_integer: bool,
}

impl VariableDefinition {
    /// Creates an unbounded continuous linear variable
    pub fn new() -> Self {
        VariableDefinition {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            initial: None,
            name: String::new(),
            is_integer: false,
        }
    }

    /// Define the variable as an integer.
    /// The variable will only be able to take an integer value in the solution.
    ///
    /// **Warning**: not all solvers support integer variables.
    /// Refer to the documentation of the solver you are using.
    ///
    /// ```
    /// # use good_lp::{ProblemVariables, variable, default_solver, SolverModel, Solution};
    /// let mut problem = ProblemVariables::new();
    /// let x = problem.add(variable().integer().min(0).max(2.5));
    /// # if cfg!(not(any(feature="clarabel"))) {
    /// let solution = problem.maximise(x).using(default_solver).solve().unwrap();
    /// // x is bound to [0; 2.5], but the solution is x=2 because x needs to be an integer
    /// assert_eq!(solution.value(x), 2.);
    /// # }
    /// ```
    pub fn integer(mut self) -> Self {
        self.is_integer = true;
        self
    }

    /// Returns `true` the variable has an integer constraint, and `false`
    /// otherwise. Note that an integer constraint can be set by adding either
    /// an integer or a binary constraint.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.is_integer(), false);
    /// let var = var.binary();
    /// assert_eq!(var.is_integer(), true);
    /// ```
    pub fn is_integer(&self) -> bool {
        self.is_integer
    }

    /// Define the variable as an integer that can only take the value 0 or 1.
    ///
    /// **Warning**: not all solvers support integer variables.
    /// Refer to the documentation of the solver you are using.
    ///
    /// ```
    /// # use good_lp::{ProblemVariables, variable, default_solver, SolverModel, Solution};
    /// let mut problem = ProblemVariables::new();
    /// let x = problem.add(variable().binary());
    /// let y = problem.add(variable().binary());
    /// if cfg!(not(any(feature="clarabel"))) {
    ///     let solution = problem.maximise(x + y).using(default_solver).solve().unwrap();
    ///     assert_eq!(solution.value(x), 1.);
    ///     assert_eq!(solution.value(y), 1.);
    /// }
    /// ```
    pub fn binary(mut self) -> Self {
        self.is_integer = true;
        self.min = 0.;
        self.max = 1.;
        self
    }

    /// Set the initial value of the variable. This may help the solver to find
    /// a solution significantly faster.
    ///
    /// Note that the given value may be disregarded by some solvers if
    /// `with_initial_solution` is called on the problem instance. It is
    /// advisable to rely either on `with_initial_solution`, or to set initial
    /// values on the variables directly, but not both.
    ///
    /// **Warning**: not all solvers support initial solutions.
    /// Refer to the documentation of the solver you are using.
    ///
    /// ```
    /// # use good_lp::{ProblemVariables, variable, default_solver, SolverModel, Solution};
    /// let mut problem = ProblemVariables::new();
    /// let x = problem.add(variable().max(3).initial(3));
    /// let y = problem.add(variable().max(5).initial(5));
    /// if cfg!(not(any(feature="clarabel"))) {
    ///     let solution = problem.maximise(x + y).using(default_solver).solve().unwrap();
    ///     assert_eq!(solution.value(x), 3.);
    ///     assert_eq!(solution.value(y), 5.);
    /// }
    /// ```
    pub fn initial<N: Into<f64>>(mut self, value: N) -> Self {
        self.initial = Some(value.into());
        self
    }

    /// Returns the optional initial value of this variable.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.get_initial(), None);
    /// let var = var.initial(42);
    /// assert_eq!(var.get_initial(), Some(42.0));
    /// ```
    pub fn get_initial(&self) -> Option<f64> {
        self.initial
    }

    /// Returns `true` the variable has an initial solution, and `false`
    /// otherwise.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.has_initial(), false);
    /// let var = var.initial(42);
    /// assert_eq!(var.has_initial(), true);
    /// ```
    pub fn has_initial(&self) -> bool {
        self.initial.is_some()
    }

    /// Set the name of the variable. This is useful in particular when displaying the problem
    /// for debugging purposes.
    ///
    /// ```
    /// # use good_lp::{ProblemVariables, variable};
    /// let mut pb = ProblemVariables::new();
    /// let x = pb.add(variable().name("x"));
    /// assert_eq!("x", pb.display(&x).to_string());
    /// ```
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// Returns the name of the variable.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.get_name(), "");
    /// let var = var.name("my variable");
    /// assert_eq!(var.get_name(), "my variable");
    /// ```
    pub fn get_name(&self) -> &str {
        &self.name
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

    /// Returns the minimum bound of the variable.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.get_min(), f64::NEG_INFINITY);
    /// let var = var.min(0);
    /// assert_eq!(var.get_min(), 0.0);
    /// ```
    pub fn get_min(&self) -> f64 {
        self.min
    }

    /// Set the higher bound of the variable
    pub fn max<N: Into<f64>>(mut self, max: N) -> Self {
        self.max = max.into();
        self
    }

    /// Returns the maximum bound of the variable.
    ///
    /// ```
    /// # use good_lp::variable;
    /// let var = variable();
    /// assert_eq!(var.get_max(), f64::INFINITY);
    /// let var = var.max(0);
    /// assert_eq!(var.get_max(), 0.0);
    /// ```
    pub fn get_max(&self) -> f64 {
        self.max
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
/// Instances of this type should be created exclusively using the [crate::variables!] macro.
#[derive(Default, Clone)]
pub struct ProblemVariables {
    variables: Vec<VariableDefinition>,
    initial_count: usize,
}

impl ProblemVariables {
    /// Create an empty list of variables
    pub fn new() -> Self {
        ProblemVariables {
            variables: vec![],
            initial_count: 0,
        }
    }

    /// Add a anonymous unbounded continuous variable to the problem
    pub fn add_variable(&mut self) -> Variable {
        self.add(variable())
    }

    /// Add a variable with the given definition
    ///
    /// ```
    /// # use good_lp::*;
    /// variables!{problem: y >= 0;}
    /// ```
    /// is equivalent to
    /// ```
    /// # use good_lp::*;
    /// let mut problem = ProblemVariables::new();
    /// let y = problem.add(variable().min(0));
    /// ```
    pub fn add(&mut self, var_def: VariableDefinition) -> Variable {
        let index = self.variables.len();
        if var_def.initial.is_some() {
            self.initial_count += 1;
        }
        self.variables.push(var_def);
        Variable::at(index)
    }

    /// Adds many variables with the given definitions
    ///
    /// ```
    /// use good_lp::*;
    /// // Solve a problem with 11 variables: x, y0, y1, ..., y9
    /// variables!{problem: 2 <= x <= 3;}
    /// let vars = vec![variable().min(0); 10];
    /// let y: Vec<Variable> = problem.add_all(vars);
    /// let objective: Expression = y.iter().sum(); // Minimise sum(y_i for i in [0; 9])
    /// let mut model = problem.minimise(objective).using(default_solver);
    /// // for all i, we must have y_i >= x
    /// for y_i in y.iter() {
    ///   model = model.with(constraint!(y_i >= x));
    /// }
    /// let solution = model.solve().unwrap();
    /// # use float_eq::assert_float_eq;
    /// assert_float_eq!(solution.value(y[3]), 2., abs <= 1e-8);
    /// ```
    pub fn add_all<T, B: FromIterator<Variable>>(&mut self, var_defs: T) -> B
    where
        T: IntoIterator<Item = VariableDefinition>,
    {
        let index = self.variables.len();
        self.variables.extend(var_defs);
        self.initial_count += self.variables[index..]
            .iter()
            .filter(|v| v.initial.is_some())
            .count();
        (index..self.variables.len()).map(Variable::at).collect()
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
    /// # use float_eq::assert_float_eq;
    /// assert_float_eq!(solution.value(y[3]), 2., abs <= 1e-8);
    /// ```
    pub fn add_vector(&mut self, var_def: VariableDefinition, len: usize) -> Vec<Variable> {
        self.add_all(repeat(var_def).take(len))
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
    ///
    /// # use float_eq::assert_float_eq;
    /// assert_float_eq!(solve(ObjectiveDirection::Minimisation), 2., abs<=1e-8);
    /// assert_float_eq!(solve(ObjectiveDirection::Maximisation), 3., abs<=1e-8);
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
    /// # use float_eq::assert_float_eq;
    /// assert_float_eq!(solution.value(x), 7., abs <= 1e-8);
    /// ```
    pub fn maximise<E: IntoAffineExpression>(self, objective: E) -> UnsolvedProblem {
        self.optimise(ObjectiveDirection::Maximisation, objective)
    }

    /// Creates an minimization problem with the given objective. Don't solve it immediately
    /// ```
    /// use good_lp::{variables, variable, default_solver, SolverModel, Solution};
    /// variables!{problem: x >= -8;}
    /// let solution = problem.minimise(x).using(default_solver).solve().unwrap();
    /// # use float_eq::assert_float_eq;
    /// assert_float_eq!(solution.value(x), -8., abs <= 1e-8);
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

    /// Returns the number of variables with initial solution values
    ///
    /// ```
    /// use good_lp::{variable, variables};
    /// let mut vars = variables!();
    /// vars.add(variable());
    /// vars.add(variable().initial(5));
    /// vars.add(variable());
    /// assert_eq!(vars.initial_solution_len(), 1);
    /// ```
    pub fn initial_solution_len(&self) -> usize {
        self.initial_count
    }

    /// Display the given expression or constraint with the correct variable names
    ///
    /// ```
    /// use good_lp::variables;
    /// variables! {problem: 0 <= x; 0 <= y;}
    /// let expression = x + 2*y;
    /// let str = problem.display(&expression).to_string();
    /// assert!(str == "x + 2 y" || str == "2 y + x"); // The ordering is not guaranteed
    /// ```
    pub fn display<'a, V: FormatWithVars>(&'a self, value: &'a V) -> impl Display + 'a {
        DisplayExpr {
            problem: self,
            value,
        }
    }
}

struct DisplayExpr<'a, 'b, V> {
    problem: &'a ProblemVariables,
    value: &'b V,
}

impl<V: FormatWithVars> Display for DisplayExpr<'_, '_, V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.value.format_with(f, |f, var| {
            let mut name = &self.problem.variables[var.index].name;
            let alternative_name: String;
            if name.is_empty() {
                alternative_name = format!("v{}", var.index);
                name = &alternative_name;
            }
            write!(f, "{}", name)
        })
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
    pub fn using<S: Solver>(self, mut solver: S) -> S::Model {
        solver.create_model(self)
    }
}

impl<N: Into<f64>> Mul<N> for Variable {
    type Output = Expression;

    fn mul(self, rhs: N) -> Self::Output {
        let mut coefficients = HashMap::with_capacity_and_hasher(1, Default::default());
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
        let mut coefficients = HashMap::with_capacity_and_hasher(1, Default::default());
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

/// Useful for binary variables. `!x` is equivalent to `1-x`.
///
/// ```
/// #[cfg(any(feature = "coin_cbc", feature = "lpsolve"))] {
/// use good_lp::*;
/// variables! {pb: x (binary); y (binary); }
/// let solution = pb.maximise(!x + y)
///                 .using(default_solver)
///                 .solve().unwrap();
/// assert_eq!(solution.value(x), 0.);
/// assert_eq!(solution.value(y), 1.);
/// # }
/// ```
impl Not for Variable {
    type Output = Expression;

    fn not(self) -> Self::Output {
        1. - self
    }
}
