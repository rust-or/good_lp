//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc).
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use std::convert::TryInto;

use coin_cbc::{raw::Status, Col, Model, Sense, Solution as CbcSolution};

use crate::solvers::{
    MipGapError, ModelWithSOS1, SolutionStatus, WithInitialSolution, WithMipGap, WithTimeLimit,
};
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
    IntoAffineExpression,
};
use crate::{Constraint, Variable};

/// The Cbc [COIN-OR](https://www.coin-or.org/) solver library.
/// To be passed to [`UnsolvedProblem::using`](crate::variable::UnsolvedProblem::using)
pub fn coin_cbc(to_solve: UnsolvedProblem) -> CoinCbcProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut model = Model::default();
    let mut initial_solution = Vec::with_capacity(variables.initial_solution_len());
    let columns: Vec<Col> = variables
        .iter_variables_with_def()
        .map(
            |(
                var,
                &VariableDefinition {
                    min,
                    max,
                    initial,
                    is_integer,
                    ..
                },
            )| {
                let col = model.add_col();
                // Variables are created with a default min of 0
                model.set_col_lower(col, min);
                if max < f64::INFINITY {
                    model.set_col_upper(col, max)
                }
                if is_integer {
                    model.set_integer(col);
                }
                if let Some(val) = initial {
                    initial_solution.push((var, val));
                };
                col
            },
        )
        .collect();
    for (var, coeff) in objective.linear.coefficients.into_iter() {
        model.set_obj_coeff(columns[var.index()], coeff);
    }
    model.set_obj_sense(match direction {
        ObjectiveDirection::Maximisation => Sense::Maximize,
        ObjectiveDirection::Minimisation => Sense::Minimize,
    });
    let mut problem = CoinCbcProblem {
        model,
        columns,
        has_sos: false,
        mip_gap: None,
    };
    if !initial_solution.is_empty() {
        problem = problem.with_initial_solution(initial_solution);
    }
    problem
}

/// A coin-cbc model
#[derive(Clone, Default)]
pub struct CoinCbcProblem {
    model: Model,
    columns: Vec<Col>,
    has_sos: bool,
    mip_gap: Option<f32>,
}

impl CoinCbcProblem {
    /// Get the inner coin_cbc model
    pub fn as_inner(&self) -> &Model {
        &self.model
    }

    /// Get a mutable version of the inner Coin CBC model.
    /// good_lp will crash (but should stay memory-safe) if you change the structure of the problem
    /// itself using this method.
    pub fn as_inner_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Set an option in cbc. For the list of available options, start the cbc binary and type '?'
    /// ```
    /// use good_lp::*;
    /// variables!{ vars: 0<=x<=1; 0<=y<=1; }
    /// let mut model = vars.maximise(x + y).using(coin_cbc);
    /// model.set_parameter("log", "1"); // Pass parameters directly to cbc
    /// let result = model.with(constraint!(x + y <= 0.5)).solve();
    /// assert_eq!(result.unwrap().value(x), 0.5);
    /// ```
    pub fn set_parameter(&mut self, key: &str, value: &str) {
        self.model.set_parameter(key, value);
    }
}

impl SolverModel for CoinCbcProblem {
    type Solution = CoinCbcSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        // Due to a bug in cbc, SOS constraints are only taken into account
        // if the model has at least one integer variable.
        // See: https://github.com/coin-or/Cbc/issues/376
        if self.has_sos {
            // We need to add two columns to work around yet another bug
            // See: https://github.com/coin-or/Cbc/issues/376#issuecomment-803057782
            let dummy_col1 = self.model.add_col();
            let dummy_col2 = self.model.add_col();
            self.model.set_obj_coeff(dummy_col1, 1e-6);
            self.model.set_obj_coeff(dummy_col2, 1e-6);
            self.model.set_integer(dummy_col1);
            let dummy_row = self.model.add_row();
            self.model.set_weight(dummy_row, dummy_col1, 1.);
            self.model.set_weight(dummy_row, dummy_col2, 1.);
            self.model.set_row_upper(dummy_row, 1.);
        }

        if let Some(mip_gap) = self.mip_gap {
            self.set_parameter("ratiogap", &mip_gap.to_string());
        }

        let solution = self.model.solve();
        let raw = solution.raw();
        match raw.status() {
            Status::Stopped => {
                if raw.is_seconds_limit_reached() {
                    let solution_vec = solution.raw().col_solution().into();
                    Ok(CoinCbcSolution{ status: SolutionStatus::TimeLimit, solution, solution_vec })
                } else {
                    Err(ResolutionError::Other("Stopped"))
                }
            },
            Status::Abandoned => Err(ResolutionError::Other("Abandoned")),
            Status::UserEvent => Err(ResolutionError::Other("UserEvent")),
            Status::Finished // The optimization finished, but may not have found a solution
            | Status::Unlaunched // The solver didn't have to be launched, presolve handled it
            => {
                if raw.is_continuous_unbounded() {
                    Err(ResolutionError::Unbounded)
                } else if raw.is_proven_infeasible() {
                    Err(ResolutionError::Infeasible)
                } else {
                    let solution_vec = solution.raw().col_solution().into();
                    Ok(CoinCbcSolution {
                        status: SolutionStatus::Optimal,
                        solution,
                        solution_vec,
                    })
                }
            },
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.model.num_rows().try_into().unwrap();
        let row = self.model.add_row();
        let constant = -constraint.expression.constant;
        if constraint.is_equality {
            self.model.set_row_equal(row, constant);
        } else {
            self.model.set_row_upper(row, constant);
        }
        for (var, coeff) in constraint.expression.linear.coefficients.into_iter() {
            self.model.set_weight(row, self.columns[var.index()], coeff);
        }
        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Coin Cbc"
    }
}

impl WithInitialSolution for CoinCbcProblem {
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        for (var, val) in solution {
            self.model
                .set_col_initial_solution(self.columns[var.index()], val);
        }
        self
    }
}

impl WithTimeLimit for CoinCbcProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.model
            .set_parameter("sec", &(seconds.into().ceil() as usize).to_string());
        self
    }
}

/// Unfortunately, the current version of cbc silently ignores
/// sos constraints on continuous variables.
/// See <https://github.com/coin-or/Cbc/issues/376>
impl ModelWithSOS1 for CoinCbcProblem {
    fn add_sos1<I: IntoAffineExpression>(&mut self, variables: I) {
        let columns = std::mem::take(&mut self.columns);
        let cols_and_weights = variables
            .linear_coefficients()
            .into_iter()
            .map(|(var, weight)| (columns[var.index()], weight));
        self.model.add_sos1(cols_and_weights);
        self.columns = columns;
        self.has_sos = true;
    }
}

/// A coin-cbc problem solution
pub struct CoinCbcSolution {
    status: SolutionStatus,
    solution: CbcSolution,
    solution_vec: Vec<f64>, // See: rust-or/good_lp#6
}

impl CoinCbcSolution {
    /// Returns the inner Coin-Cbc model
    pub fn model(&self) -> &coin_cbc::raw::Model {
        self.solution.raw()
    }
}

impl Solution for CoinCbcSolution {
    fn status(&self) -> SolutionStatus {
        self.status
    }
    fn value(&self, variable: Variable) -> f64 {
        // Our indices should always match those of cbc
        self.solution_vec[variable.index()]
    }
}

impl WithMipGap for CoinCbcProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.mip_gap
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        if mip_gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if mip_gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            self.mip_gap = Some(mip_gap);
            Ok(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::coin_cbc;
    use crate::{
        solvers::{SolutionStatus, WithTimeLimit},
        variable, variables, Expression, Solution, SolverModel, WithInitialSolution,
    };
    use float_eq::assert_float_eq;

    #[test]
    fn solve_problem_with_time_limit() {
        let n = 10;
        let mut vars = variables!();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(vars.add(variable().binary()));
        }
        let pb = vars
            .maximise(v.iter().map(|&v| 3.5 * v).sum::<Expression>())
            .using(coin_cbc)
            .with_time_limit(0.0);
        let sol = pb.solve().unwrap();
        assert!(matches!(sol.status(), SolutionStatus::TimeLimit));
        for var in v {
            assert_eq!(sol.value(var), 1.0);
        }
    }

    #[test]
    fn solve_problem_with_initial_solution() {
        let limit = 3.0;
        // Solve problem once
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
        // Recreate problem and solve with initial solution
        let initial_solution = vec![(v, sol.value(v))];
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars
            .maximise(v)
            .using(coin_cbc)
            .with_initial_solution(initial_solution);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
    }

    #[test]
    fn solve_problem_with_initial_variable_values() {
        let limit = 3.0;
        // Solve problem once
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
        // Recreate problem and solve with initial solution
        let mut vars = variables!();
        let v = vars.add(variable().min(0).max(limit).initial(2));
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
    }
}
