//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc).
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use crate::solvers::ModelWithSOS1;
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
    IntoAffineExpression,
};
use crate::{Constraint, Variable};
use coin_cbc::{raw::Status, Col, Model, Sense, Solution as CbcSolution};
use std::convert::TryInto;

/// The Cbc [COIN-OR](https://www.coin-or.org/) solver library.
/// To be passed to [`UnsolvedProblem::using`](crate::variable::UnsolvedProblem::using)
pub fn coin_cbc(to_solve: UnsolvedProblem) -> CoinCbcProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut model = Model::default();
    let columns: Vec<Col> = variables
        .into_iter()
        .map(|VariableDefinition { min, max, is_integer, .. }| {
            let col = model.add_col();
            // Variables are created with a default min of 0
            model.set_col_lower(col, min);
            if max < f64::INFINITY {
                model.set_col_upper(col, max)
            }
            if is_integer {
                model.set_integer(col);
            }
            col
        })
        .collect();
    for (var, coeff) in objective.linear.coefficients.into_iter() {
        model.set_obj_coeff(columns[var.index()], coeff);
    }
    model.set_obj_sense(match direction {
        ObjectiveDirection::Maximisation => Sense::Maximize,
        ObjectiveDirection::Minimisation => Sense::Minimize,
    });
    CoinCbcProblem {
        model,
        columns,
        has_sos: false,
    }
}

/// A coin-cbc model
pub struct CoinCbcProblem {
    model: Model,
    columns: Vec<Col>,
    has_sos: bool,
}

impl CoinCbcProblem {
    /// Get the inner coin_cbc model
    pub fn as_inner(&self) -> &Model {
        &self.model
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

        let solution = self.model.solve();
        let raw = solution.raw();
        match raw.status() {
            Status::Stopped => Err(ResolutionError::Other("Stopped")),
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
                    Ok(CoinCbcSolution {
                        columns: self.columns,
                        solution,
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
    columns: Vec<Col>,
    solution: CbcSolution,
}

impl CoinCbcSolution {
    /// Returns the inner Coin-Cbc model
    pub fn model(&self) -> &coin_cbc::raw::Model {
        self.solution.raw()
    }
}

impl Solution for CoinCbcSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution.col(self.columns[variable.index()])
    }
}
