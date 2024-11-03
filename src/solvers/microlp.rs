//! A solver that uses [microlp](https://docs.rs/microlp), a pure rust solver.

use std::panic::catch_unwind;

use microlp::Error;

use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};

/// The [microlp](https://docs.rs/microlp) solver,
/// to be used with [UnsolvedProblem::using].
pub fn microlp(to_solve: UnsolvedProblem) -> MicroLpProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut problem = microlp::Problem::new(match direction {
        ObjectiveDirection::Maximisation => microlp::OptimizationDirection::Maximize,
        ObjectiveDirection::Minimisation => microlp::OptimizationDirection::Minimize,
    });
    let mut integers: Vec<microlp::Variable> = vec![];
    let variables: Vec<microlp::Variable> = variables
        .iter_variables_with_def()
        .map(
            |(
                var,
                &VariableDefinition {
                    min,
                    max,
                    is_integer,
                    ..
                },
            )| {
                let coeff = *objective.linear.coefficients.get(&var).unwrap_or(&0.);
                let var = problem.add_var(coeff, (min, max));
                if is_integer {
                    integers.push(var);
                }
                var
            },
        )
        .collect();
    MicroLpProblem {
        problem,
        variables,
        integers,
        n_constraints: 0,
    }
}

/// A microlp model
pub struct MicroLpProblem {
    problem: microlp::Problem,
    variables: Vec<microlp::Variable>,
    integers: Vec<microlp::Variable>,
    n_constraints: usize,
}

impl MicroLpProblem {
    /// Get the inner microlp model
    pub fn as_inner(&self) -> &microlp::Problem {
        &self.problem
    }
}

impl SolverModel for MicroLpProblem {
    type Solution = MicroLpSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let mut solution = self.problem.solve()?;
        for int_var in self.integers {
            solution = catch_unwind(|| solution.add_gomory_cut(int_var)).map_err(|_| {
                ResolutionError::Other("microlp does not support integer variables")
            })??;
        }
        Ok(MicroLpSolution {
            solution,
            variables: self.variables,
        })
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.n_constraints;
        let op = match constraint.is_equality {
            true => microlp::ComparisonOp::Eq,
            false => microlp::ComparisonOp::Le,
        };
        let constant = -constraint.expression.constant;
        let mut linear_expr = microlp::LinearExpr::empty();
        for (var, coefficient) in constraint.expression.linear.coefficients {
            linear_expr.add(self.variables[var.index()], coefficient);
        }
        self.problem.add_constraint(linear_expr, op, constant);
        self.n_constraints += 1;
        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Microlp"
    }
}

impl From<microlp::Error> for ResolutionError {
    fn from(microlp_error: Error) -> Self {
        match microlp_error {
            microlp::Error::Unbounded => Self::Unbounded,
            microlp::Error::Infeasible => Self::Infeasible,
            microlp::Error::InternalError(s) => Self::Str(s),
        }
    }
}

/// The solution to a microlp problem
pub struct MicroLpSolution {
    solution: microlp::Solution,
    variables: Vec<microlp::Variable>,
}

impl MicroLpSolution {
    /// Returns the MicroLP solution object. You can use it to dynamically add new constraints
    pub fn into_inner(self) -> microlp::Solution {
        self.solution
    }
}

impl Solution for MicroLpSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution[self.variables[variable.index()]]
    }
}

#[cfg(test)]
mod tests {
    use crate::{variable, variables, Solution, SolverModel};

    use super::microlp;

    #[test]
    fn can_solve_easy() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(microlp)
            .with((2 * x + y) << 4)
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (0.5, 3.))
    }
}
