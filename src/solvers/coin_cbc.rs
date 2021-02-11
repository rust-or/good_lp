use crate::solvers::{Solver, ObjectiveDirection, Solution, ResolutionError};
use crate::{Constraint, Expression, Variable};
use crate::variable::ProblemVariables;
use coin_cbc::{raw::Status, Model, Sense, Col, Solution as CbcSolution};
use std::marker::PhantomData;

struct CoinCbc<F> {
    model: Model,
    columns: Vec<Col>,
    variable_type: PhantomData<F>,
}

impl<T> Solver<T> for CoinCbc<T> {
    type Solution = CoinCbcSolution<T>;
    type Error = ResolutionError;

    fn new(variables: ProblemVariables<T>, direction: ObjectiveDirection, objective: Expression<T>) -> Self {
        let mut model = Model::default();
        let columns: Vec<Col> = variables.into_iter().map(
            |_var| model.add_col()
        ).collect();
        for (var, coeff) in objective.linear.coefficients.into_iter() {
            model.set_obj_coeff(columns[var.index()], coeff);
        }
        model.set_obj_sense(match direction {
            ObjectiveDirection::Maximisation => Sense::Maximize,
            ObjectiveDirection::Minimisation => Sense::Minimize,
        });
        CoinCbc { model, columns, variable_type: PhantomData }
    }

    fn with(&mut self, constraint: Constraint<T>) -> &mut Self {
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
        if self.model.to_raw().is_continuous_unbounded() {
            return Err(ResolutionError::Unbounded);
        }
        match solution.raw().status() {
            Status::Unlaunched => Err(ResolutionError::Other("Unlaunched")),
            Status::Stopped => Err(ResolutionError::Other("Stopped")),
            Status::Abandoned => Err(ResolutionError::Other("Abandoned")),
            Status::UserEvent => Err(ResolutionError::Other("UserEvent")),
            Status::Finished => Ok(CoinCbcSolution {
                columns: self.columns,
                solution,
                variable_type: PhantomData,
            }),
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