//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc)
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use crate::solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel};
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{Constraint, Variable};
use coin_cbc::{raw::Status, Col, Model, Sense, Solution as CbcSolution};
use std::marker::PhantomData;

/// The Cbc [COIN-OR](https://www.coin-or.org/) solver library
pub fn coin_cbc<F>(to_solve: UnsolvedProblem<F>) -> CoinCbcProblem<F> {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut model = Model::default();
    let columns: Vec<Col> = variables
        .into_iter()
        .map(|VariableDefinition { min, max, .. }| {
            let col = model.add_col();
            if min > f64::NEG_INFINITY {
                model.set_col_lower(col, min)
            }
            if max < f64::INFINITY {
                model.set_col_upper(col, max)
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
        variable_type: PhantomData,
    }
}

pub struct CoinCbcProblem<F> {
    model: Model,
    columns: Vec<Col>,
    variable_type: PhantomData<F>,
}

impl<T> SolverModel<T> for CoinCbcProblem<T> {
    type Solution = CoinCbcSolution<T>;
    type Error = ResolutionError;

    fn with(mut self, constraint: Constraint<T>) -> Self {
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
        self
    }

    fn solve(self) -> Result<Self::Solution, Self::Error> {
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
                        variable_type: PhantomData,
                    })
                }
            },
        }
    }
}

pub struct CoinCbcSolution<F> {
    columns: Vec<Col>,
    solution: CbcSolution,
    variable_type: PhantomData<F>,
}

impl<F> Solution<F> for CoinCbcSolution<F> {
    fn value(&self, variable: Variable<F>) -> f64 {
        self.solution.col(self.columns[variable.index()])
    }
}
