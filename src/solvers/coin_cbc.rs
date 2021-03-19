//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc).
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};
use coin_cbc::{raw::Status, Col, Model, Sense, Solution as CbcSolution};

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
        .map(|VariableDefinition { min, max, .. }| {
            let col = model.add_col();
            // Variables are created with a default min of 0
            model.set_col_lower(col, min);
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
        n_constraints: 0,
    }
}

/// A coin-cbc model
pub struct CoinCbcProblem {
    model: Model,
    columns: Vec<Col>,
    n_constraints: usize,
}

impl CoinCbcProblem {
    /// Get the inner coin_cbc model
    pub fn as_inner(&self) -> &Model {
        &self.model
    }

    /// Default implementation for adding a constraint to the Problem
    fn put_constraint(&mut self, constraint: Constraint) {
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
        self.n_constraints += 1;
    }
}

impl SolverModel for CoinCbcProblem {
    type Solution = CoinCbcSolution;
    type Error = ResolutionError;

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
                    })
                }
            },
        }
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        self.put_constraint(c);

        ConstraintReference {
            index: self.n_constraints - 1,
        }
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
