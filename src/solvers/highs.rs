//! A solver that uses [highs](https://docs.rs/highs), a parallel C++ solver.

use highs::HighsModelStatus;

use crate::solvers::{
    ObjectiveDirection, ResolutionError, Solution, SolutionWithDual, SolverModel, WithMipGap,
};
use crate::{
    constraint::ConstraintReference,
    solvers::DualValues,
    variable::{UnsolvedProblem, VariableDefinition},
};
use crate::{Constraint, IntoAffineExpression, Variable};

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
        verbose: false,
        options: HighsOptions::default(),
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
            HighsSolverType::Simplex => "s1implex",
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

#[derive(Debug, Clone, Copy)]
struct HighsOptions {
    presolve: HighsPresolveType,
    solver: HighsSolverType,
    parallel: HighsParallelType,
    mip_abs_gap: Option<f32>,
    mip_rel_gap: Option<f32>,
    time_limit: f64,
    threads: u32,
}

impl Default for HighsOptions {
    fn default() -> Self {
        Self {
            presolve: HighsPresolveType::Choose,
            solver: HighsSolverType::Choose,
            parallel: HighsParallelType::Choose,
            mip_abs_gap: None,
            mip_rel_gap: None,
            time_limit: f64::MAX,
            threads: 0,
        }
    }
}

/// A HiGHS model
#[derive(Debug)]
pub struct HighsProblem {
    sense: highs::Sense,
    highs_problem: highs::RowProblem,
    columns: Vec<highs::Col>,
    verbose: bool,
    options: HighsOptions,
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

    /// Sets HiGHS Presolve Option
    pub fn set_presolve(mut self, presolve: HighsPresolveType) -> HighsProblem {
        self.options.presolve = presolve;
        self
    }

    /// Sets HiGHS Solver Option
    pub fn set_solver(mut self, solver: HighsSolverType) -> HighsProblem {
        self.options.solver = solver;
        self
    }

    /// Sets HiGHS Parallel Option
    pub fn set_parallel(mut self, parallel: HighsParallelType) -> HighsProblem {
        self.options.parallel = parallel;
        self
    }

    /// Sets HiGHS Tolerance on Absolute Gap Option
    pub fn set_mip_abs_gap(mut self, mip_abs_gap: f32) -> Result<HighsProblem, String> {
        if mip_abs_gap.is_sign_positive() && mip_abs_gap.is_finite() {
            self.options.mip_abs_gap = Some(mip_abs_gap);
            Ok(self)
        } else {
            Err("Invalid MIP gap: must be positive and finite".to_string())
        }
    }

    /// Sets HiGHS Tolerance on Relative Gap Option
    pub fn set_mip_rel_gap(mut self, mip_rel_gap: f32) -> Result<HighsProblem, String> {
        if mip_rel_gap.is_sign_positive() && mip_rel_gap.is_finite() {
            self.options.mip_rel_gap = Some(mip_rel_gap);
            Ok(self)
        } else {
            Err("Invalid MIP gap: must be positive and finite".to_string())
        }
    }

    /// Sets HiGHS Time Limit Option
    pub fn set_time_limit(mut self, time_limit: f64) -> HighsProblem {
        self.options.time_limit = time_limit;
        self
    }

    /// Sets number of threads used by HiGHS
    pub fn set_threads(mut self, threads: u32) -> HighsProblem {
        self.options.threads = threads;
        self
    }
}

impl SolverModel for HighsProblem {
    type Solution = HighsSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let verbose = self.verbose;
        let options = self.options;
        let mut model = self.into_inner();
        if verbose {
            model.set_option(&b"output_flag"[..], true);
            model.set_option(&b"log_to_console"[..], true);
            model.set_option(&b"log_dev_level"[..], 2);
        }
        model.set_option("presolve", options.presolve.as_str());
        model.set_option("solver", options.solver.as_str());
        model.set_option("parallel", options.parallel.as_str());

        if let Some(mip_abs_gap) = options.mip_abs_gap {
            model.set_option("mip_abs_gap", mip_abs_gap as f64);
        }

        if let Some(mip_rel_gap) = options.mip_rel_gap {
            model.set_option("mip_rel_gap", mip_rel_gap as f64);
        }

        model.set_option("time_limit", options.time_limit);
        model.set_option("threads", options.threads as i32);

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
            .into_iter()
            .map(|(variable, factor)| (columns[variable.index()], factor));
        if constraint.is_equality {
            self.highs_problem
                .add_row(upper_bound..=upper_bound, factors);
        } else {
            self.highs_problem.add_row(..=upper_bound, factors);
        }
        ConstraintReference { index }
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
        self.options.mip_rel_gap
    }

    fn with_mip_gap(self, mip_gap: f32) -> Result<Self, String> {
        self.set_mip_rel_gap(mip_gap)
    }
}
