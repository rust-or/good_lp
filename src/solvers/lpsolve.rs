//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc).
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use crate::solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel};
use crate::variable::UnsolvedProblem;
use crate::{
    affine_expression_trait::IntoAffineExpression, constraint::ConstraintReference, ModelWithSOS1,
};
use crate::{Constraint, Variable};
use lpsolve::{ConstraintType, Problem, SOSType, SolveStatus};
use std::convert::TryInto;
use std::ffi::CString;
use std::os::raw::c_int;

/// String constant storing the name of the default solver.
/// Re-exported in `src/lib.rs`.
pub const DEFAULT_SOLVER_NAME: &str = "lpsolve";

fn expr_to_scatter_vec<E: IntoAffineExpression>(expr: E) -> (Vec<f64>, Vec<c_int>, f64) {
    let constant = expr.constant();
    let mut indices: Vec<c_int> = vec![];
    let mut coefficients = vec![];
    for (var, coeff) in expr.linear_coefficients() {
        indices.push(col_num(var));
        coefficients.push(coeff);
    }
    (coefficients, indices, constant)
}

fn to_c(i: usize) -> c_int {
    i.try_into().expect("Too many variables.")
}

fn col_num(var: Variable) -> c_int {
    to_c(var.index() + 1)
}

/// The [lp_solve](http://lpsolve.sourceforge.net/5.5/) open-source solver library.
/// lp_solve is released under the LGPL license.
pub fn lp_solve(to_solve: UnsolvedProblem) -> LpSolveProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    // It looks like the lp_solve rust binding doesn't expose the set_maxim function
    let objective = if direction == ObjectiveDirection::Minimisation {
        objective
    } else {
        -objective
    };

    let cols = to_c(variables.len());
    let mut model = Problem::new(0, cols).expect("Unable to create problem");
    let (obj_coefs, obj_idx, _const) = expr_to_scatter_vec(objective);
    assert!(model.scatter_objective_function(&obj_coefs, &obj_idx));
    for (i, v) in variables.into_iter().enumerate() {
        let col = to_c(i + 1);
        assert!(model.set_integer(col, v.is_integer));
        if v.min.is_finite() || v.max.is_finite() {
            assert!(model.set_bounds(col, v.min, v.max));
        } else {
            assert!(model.set_unbounded(col));
        }
    }
    LpSolveProblem(model)
}

/// An lp_solve problem instance
pub struct LpSolveProblem(Problem);

impl SolverModel for LpSolveProblem {
    type Solution = LpSolveSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        use ResolutionError::*;
        match Problem::solve(&mut self.0) {
            SolveStatus::Unbounded => Err(Unbounded),
            SolveStatus::Infeasible => Err(Infeasible),
            SolveStatus::OutOfMemory => Err(Other("OutOfMemory")),
            SolveStatus::NotRun => Err(Other("NotRun")),
            SolveStatus::Degenerate => Err(Other("Degenerate")),
            SolveStatus::NumericalFailure => Err(Other("NumericalFailure")),
            SolveStatus::UserAbort => Err(Other("UserAbort")),
            SolveStatus::Timeout => Err(Other("Timeout")),
            SolveStatus::ProcFail => Err(Other("ProcFail")),
            SolveStatus::ProcBreak => Err(Other("ProcBreak")),
            SolveStatus::NoFeasibleFound => Err(Other("NoFeasibleFound")),
            _ => {
                let mut solution = vec![0.; self.0.num_cols() as usize];
                let truncated = self
                    .0
                    .get_solution_variables(&mut solution)
                    .expect("internal error: invalid solution array length");
                assert_eq!(
                    truncated.len(),
                    solution.len(),
                    "The solution doesn't have the expected number of variables"
                );
                Ok(LpSolveSolution {
                    problem: self.0,
                    solution,
                })
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.0.num_rows().try_into().expect("too many rows");
        let mut coeffs: Vec<f64> = vec![0.; self.0.num_cols() as usize + 1];
        let target = -constraint.expression.constant;
        for (var, coeff) in constraint.expression.linear_coefficients() {
            coeffs[var.index() + 1] = coeff;
        }
        let constraint_type = if constraint.is_equality {
            ConstraintType::Eq
        } else {
            ConstraintType::Le
        };
        let success = self.0.add_constraint(&coeffs, target, constraint_type);
        assert!(success, "could not add constraint. memory error.");
        ConstraintReference { index }
    }
}

impl ModelWithSOS1 for LpSolveProblem {
    fn add_sos1<I: IntoAffineExpression>(&mut self, variables: I) {
        let iter = variables.linear_coefficients().into_iter();
        let (len, _) = iter.size_hint();
        let mut weights = Vec::with_capacity(len);
        let mut variables = Vec::with_capacity(len);
        for (var, weight) in iter {
            weights.push(weight);
            variables.push(var.index().try_into().expect("too many vars"));
        }
        let name = CString::new("sos").unwrap();
        self.0
            .add_sos_constraint(&name, SOSType::Type1, 1, &weights, &variables);
    }
}

/// A coin-cbc problem solution
pub struct LpSolveSolution {
    problem: Problem,
    solution: Vec<f64>,
}

impl LpSolveSolution {
    /// Returns the inner Coin-Cbc model
    pub fn into_inner(self) -> Problem {
        self.problem
    }
}

impl Solution for LpSolveSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution[variable.index()]
    }
}
