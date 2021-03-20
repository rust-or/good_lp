//! A solver that uses [minilp](https://docs.rs/minilp), a pure rust solver.

use std::panic::catch_unwind;

use minilp::Error;

use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};

/// The [minilp](https://docs.rs/minilp) solver,
/// to be used with [UnsolvedProblem::using].
pub fn minilp(to_solve: UnsolvedProblem) -> MiniLpProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut problem = minilp::Problem::new(match direction {
        ObjectiveDirection::Maximisation => minilp::OptimizationDirection::Maximize,
        ObjectiveDirection::Minimisation => minilp::OptimizationDirection::Minimize,
    });
    let mut integers: Vec<minilp::Variable> = vec![];
    let variables: Vec<minilp::Variable> = variables
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
    MiniLpProblem {
        problem,
        variables,
        integers,
        n_constraints: 0,
    }
}

/// A minilp model
pub struct MiniLpProblem {
    problem: minilp::Problem,
    variables: Vec<minilp::Variable>,
    integers: Vec<minilp::Variable>,
    n_constraints: usize,
}

impl MiniLpProblem {
    /// Get the inner minilp model
    pub fn as_inner(&self) -> &minilp::Problem {
        &self.problem
    }
}

impl SolverModel for MiniLpProblem {
    type Solution = MiniLpSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let mut solution = self.problem.solve()?;
        for int_var in self.integers {
            solution = catch_unwind(|| solution.add_gomory_cut(int_var)).map_err(|_| {
                ResolutionError::Other("minilp does not support integer variables")
            })??;
        }
        Ok(MiniLpSolution {
            solution,
            variables: self.variables,
        })
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.n_constraints;
        let op = match constraint.is_equality {
            true => minilp::ComparisonOp::Eq,
            false => minilp::ComparisonOp::Le,
        };
        let constant = -constraint.expression.constant;
        let mut linear_expr = minilp::LinearExpr::empty();
        for (var, coefficient) in constraint.expression.linear.coefficients {
            linear_expr.add(self.variables[var.index()], coefficient);
        }
        self.problem.add_constraint(linear_expr, op, constant);
        self.n_constraints += 1;
        ConstraintReference { index }
    }
}

impl From<minilp::Error> for ResolutionError {
    fn from(minilp_error: Error) -> Self {
        match minilp_error {
            minilp::Error::Unbounded => Self::Unbounded,
            minilp::Error::Infeasible => Self::Infeasible,
        }
    }
}

/// The solution to a minilp problem
pub struct MiniLpSolution {
    solution: minilp::Solution,
    variables: Vec<minilp::Variable>,
}

impl MiniLpSolution {
    /// Returns the MiniLP solution object. You can use it to dynamically add new constraints
    pub fn into_inner(self) -> minilp::Solution {
        self.solution
    }
}

impl Solution for MiniLpSolution {
    fn value(&self, variable: Variable) -> f64 {
        self.solution[self.variables[variable.index()]]
    }
}

#[cfg(test)]
mod tests {
    use crate::{variable, variables, Solution, SolverModel};

    use super::minilp;

    #[test]
    fn can_solve_easy() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(minilp)
            .with((2 * x + y) << 4)
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (0.5, 3.))
    }
}
