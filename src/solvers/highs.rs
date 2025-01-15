//! A solver that uses [highs](https://docs.rs/highs), a parallel C++ solver.

use crate::solvers::{
    MipGapError, ObjectiveDirection, ResolutionError, Solution, SolutionWithDual, SolverModel,
    WithMipGap,
};
use crate::{
    constraint::ConstraintReference,
    solvers::DualValues,
    variable::{UnsolvedProblem, VariableDefinition},
};
use crate::{Constraint, IntoAffineExpression, Variable, WithInitialSolution};
use highs::HighsModelStatus;
use std::collections::HashMap;
use std::iter::FromIterator;

/// The [highs](https://docs.rs/highs) solver,
/// to be used with [UnsolvedProblem::using].
///
/// This solver does not support integer variables and will panic
/// if given a problem with integer variables.
pub fn highs(to_solve: UnsolvedProblem) -> HighsProblem {
    let mut highs_problem = highs::RowProblem::default();
    let sense = match to_solve.direction {
        ObjectiveDirection::Maximisation => highs::Sense::Maximise,
        ObjectiveDirection::Minimisation => highs::Sense::Minimise,
    };
    let mut columns = Vec::with_capacity(to_solve.variables.len());
    for (
        var,
        &VariableDefinition {
            min,
            max,
            is_integer,
            ..
        },
    ) in to_solve.variables.iter_variables_with_def()
    {
        let &col_factor = to_solve
            .objective
            .linear
            .coefficients
            .get(&var)
            .unwrap_or(&0.);
        let col = highs_problem.add_column_with_integrality(col_factor, min..max, is_integer);
        columns.push(col);
    }
    HighsProblem {
        sense,
        highs_problem,
        columns,
        initial_solution: None,
        verbose: false,
        options: Default::default(),
    }
}

/// Presolve option
#[derive(Debug, Clone, Copy)]
pub enum HighsPresolveType {
    /// off
    Off,
    /// choose
    Choose,
    /// on
    On,
}

impl HighsPresolveType {
    fn as_str(&self) -> &str {
        match self {
            HighsPresolveType::Off => "off",
            HighsPresolveType::Choose => "choose",
            HighsPresolveType::On => "on",
        }
    }
}

/// Solver option
#[derive(Debug, Clone, Copy)]
pub enum HighsSolverType {
    /// simplex
    Simplex,
    /// choose
    Choose,
    /// ipm
    Ipm,
}

impl HighsSolverType {
    fn as_str(&self) -> &str {
        match self {
            HighsSolverType::Simplex => "simplex",
            HighsSolverType::Choose => "choose",
            HighsSolverType::Ipm => "ipm",
        }
    }
}

/// Parallel option
#[derive(Debug, Clone, Copy)]
pub enum HighsParallelType {
    /// off
    Off,
    /// choose
    Choose,
    /// on
    On,
}

impl HighsParallelType {
    fn as_str(&self) -> &str {
        match self {
            HighsParallelType::Off => "off",
            HighsParallelType::Choose => "choose",
            HighsParallelType::On => "on",
        }
    }
}

/// A HiGHS option value.
#[derive(Debug, Clone)]
pub enum HighsOptionValue {
    /// String option
    String(String),
    /// Boolean option
    Bool(bool),
    /// Integer option
    Int(i32),
    /// Floating point number option
    Float(f64),
}
impl HighsOptionValue {
    /// Gets the float option if applicable.
    pub fn as_float(&self) -> Option<f64> {
        if let &Self::Float(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
impl From<bool> for HighsOptionValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i32> for HighsOptionValue {
    fn from(v: i32) -> Self {
        Self::Int(v)
    }
}
impl From<f64> for HighsOptionValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<String> for HighsOptionValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<&str> for HighsOptionValue {
    fn from(v: &str) -> Self {
        Self::String(v.into())
    }
}

/// A HiGHS model
#[derive(Debug)]
pub struct HighsProblem {
    sense: highs::Sense,
    highs_problem: highs::RowProblem,
    columns: Vec<highs::Col>,
    initial_solution: Option<Vec<(Variable, f64)>>,
    verbose: bool,
    options: HashMap<String, HighsOptionValue>,
}

impl HighsProblem {
    /// Get a highs model for this problem
    pub fn into_inner(self) -> highs::Model {
        self.highs_problem.optimise(self.sense)
    }

    /// Sets whether or not HiGHS should display verbose logging information to the console
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose
    }

    /// Sets the HiGHS option. See https://ergo-code.github.io/HiGHS/dev/options/definitions/
    pub fn set_option<K: Into<String>, V: Into<HighsOptionValue>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Sets HiGHS Presolve Option
    pub fn set_presolve(self, presolve: HighsPresolveType) -> HighsProblem {
        self.set_option("presolve", presolve.as_str())
    }

    /// Sets HiGHS Solver Option
    pub fn set_solver(self, solver: HighsSolverType) -> HighsProblem {
        self.set_option("solver", solver.as_str())
    }

    /// Sets HiGHS Parallel Option
    pub fn set_parallel(self, parallel: HighsParallelType) -> HighsProblem {
        self.set_option("parallel", parallel.as_str())
    }

    /// Sets HiGHS Tolerance on Absolute Gap Option
    pub fn set_mip_abs_gap(self, mip_abs_gap: f32) -> Result<HighsProblem, MipGapError> {
        if mip_abs_gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if mip_abs_gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            Ok(self.set_option("mip_abs_gap", mip_abs_gap as f64))
        }
    }

    /// Sets HiGHS Tolerance on Relative Gap Option
    pub fn set_mip_rel_gap(self, mip_rel_gap: f32) -> Result<HighsProblem, MipGapError> {
        if mip_rel_gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if mip_rel_gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            Ok(self.set_option("mip_rel_gap", mip_rel_gap as f64))
        }
    }

    /// Sets HiGHS Time Limit Option
    pub fn set_time_limit(self, time_limit: f64) -> HighsProblem {
        self.set_option("time_limit", time_limit)
    }

    /// Sets number of threads used by HiGHS
    pub fn set_threads(self, threads: u32) -> HighsProblem {
        self.set_option("threads", threads as i32)
    }
}

impl SolverModel for HighsProblem {
    type Solution = HighsSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        let verbose = self.verbose;
        let options = std::mem::take(&mut self.options);
        let initial_solution = self.initial_solution.as_ref().map(|pairs| {
            pairs
                .iter()
                .fold(vec![0.0; self.columns.len()], |mut sol, (var, val)| {
                    sol[var.index()] = *val;
                    sol
                })
        });

        let mut model = self.into_inner();
        if verbose {
            model.set_option(&b"output_flag"[..], true);
            model.set_option(&b"log_to_console"[..], true);
            model.set_option(&b"log_dev_level"[..], 2);
        }
        for (k, v) in options {
            match v {
                HighsOptionValue::String(v) => model.set_option(k, v.as_str()),
                HighsOptionValue::Float(v) => model.set_option(k, v),
                HighsOptionValue::Bool(v) => model.set_option(k, v),
                HighsOptionValue::Int(v) => model.set_option(k, v),
            }
        }

        if initial_solution.is_some() {
            model.set_solution(initial_solution.as_deref(), None, None, None);
        }

        let solved = model.solve();
        match solved.status() {
            HighsModelStatus::NotSet => Err(ResolutionError::Other("NotSet")),
            HighsModelStatus::LoadError => Err(ResolutionError::Other("LoadError")),
            HighsModelStatus::ModelError => Err(ResolutionError::Other("ModelError")),
            HighsModelStatus::PresolveError => Err(ResolutionError::Other("PresolveError")),
            HighsModelStatus::SolveError => Err(ResolutionError::Other("SolveError")),
            HighsModelStatus::PostsolveError => Err(ResolutionError::Other("PostsolveError")),
            HighsModelStatus::ModelEmpty => Err(ResolutionError::Other("ModelEmpty")),
            HighsModelStatus::Infeasible => Err(ResolutionError::Infeasible),
            HighsModelStatus::Unbounded => Err(ResolutionError::Unbounded),
            HighsModelStatus::UnboundedOrInfeasible => Err(ResolutionError::Infeasible),
            _ok_status => Ok(HighsSolution {
                solution: solved.get_solution(),
            }),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.highs_problem.num_rows();
        let upper_bound = -constraint.expression.constant();
        let columns = &self.columns;
        let factors = constraint
            .expression
            .linear_coefficients()
            .map(|(variable, factor)| (columns[variable.index()], factor));
        if constraint.is_equality {
            self.highs_problem
                .add_row(upper_bound..=upper_bound, factors);
        } else {
            self.highs_problem.add_row(..=upper_bound, factors);
        }
        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Highs"
    }
}

impl WithInitialSolution for HighsProblem {
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        self.initial_solution = Some(Vec::from_iter(solution));
        self
    }
}

/// The solution to a highs problem
#[derive(Debug)]
pub struct HighsSolution {
    solution: highs::Solution,
}

impl HighsSolution {
    /// Returns the highs solution object. You can use it to fetch dual values
    pub fn into_inner(self) -> highs::Solution {
        self.solution
    }
}

impl Solution for HighsSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution.columns()[variable.index()]
    }
}

impl<'a> DualValues for &'a HighsSolution {
    fn dual(&self, constraint: ConstraintReference) -> f64 {
        self.solution.dual_rows()[constraint.index]
    }
}

impl<'a> SolutionWithDual<'a> for HighsSolution {
    type Dual = &'a HighsSolution;

    fn compute_dual(&'a mut self) -> &'a HighsSolution {
        self
    }
}

impl WithMipGap for HighsProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.options
            .get("mip_rel_gap")?
            .as_float()
            .map(|v| v as f32)
    }

    fn with_mip_gap(self, mip_gap: f32) -> Result<Self, MipGapError> {
        self.set_mip_rel_gap(mip_gap)
    }
}
