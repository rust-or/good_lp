//! A solver that uses [highs](https://docs.rs/highs), a parallel C++ solver.

use highs::HighsModelStatus;

use crate::solvers::{
    ObjectiveDirection, ResolutionError, Solution, SolutionWithDual, SolverModel,
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
        if is_integer {
            panic!("HiGHS does not support integer variables, but variable number {} is of type integer.", var.index());
        }
        let &col_factor = to_solve
            .objective
            .linear
            .coefficients
            .get(&var)
            .unwrap_or(&0.);
        let col = highs_problem.add_column(col_factor, min..max);
        columns.push(col);
    }
    HighsProblem {
        sense,
        highs_problem,
        columns,
        verbose: false,
    }
}

/// A HiGHS model
#[derive(Debug)]
pub struct HighsProblem {
    sense: highs::Sense,
    highs_problem: highs::RowProblem,
    columns: Vec<highs::Col>,
    verbose: bool,
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
}

impl SolverModel for HighsProblem {
    type Solution = HighsSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let verbose = self.verbose;
        let mut model = self.into_inner();
        if verbose {
            model.set_option(&b"output_flag"[..], true);
            model.set_option(&b"log_to_console"[..], true);
            model.set_option(&b"log_dev_level"[..], 2);
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
                dual_values: vec![],
                acquired: false,
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
    dual_values: Vec<f64>,
    acquired: bool,
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
