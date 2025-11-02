//! A solver that uses [microlp](https://docs.rs/microlp), a pure rust solver.

use microlp::Error;

use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolutionStatus, SolverModel},
    WithTimeLimit,
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
                if is_integer {
                    problem.add_integer_var(coeff, (min as i32, max as i32))
                } else {
                    problem.add_var(coeff, (min, max))
                }
            },
        )
        .collect();
    MicroLpProblem {
        problem,
        variables,
        n_constraints: 0,
    }
}

/// A microlp model
pub struct MicroLpProblem {
    problem: microlp::Problem,
    variables: Vec<microlp::Variable>,
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
        let solution = self.problem.solve()?;
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

impl WithTimeLimit for MicroLpProblem {
    fn with_time_limit<T: Into<f64>>(self, _seconds: T) -> Self {
        // microlp does not support time limits yet
        self
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
    fn status(&self) -> SolutionStatus {
        SolutionStatus::Optimal
    }
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

    #[test]
    fn can_solve_milp() {
        let mut vars = variables!();

        let x = vars.add(variable().clamp(2, f64::INFINITY));
        let y = vars.add(variable().clamp(0, 7));
        let z = vars.add(variable().integer().clamp(0, f64::INFINITY));

        let solution = vars
            .maximise(50 * x + 40 * y + 45 * z)
            .using(microlp)
            .with((3 * x + 2 * y + z) << 20)
            .with((2 * x + y + 3 * z) << 15)
            .solve()
            .unwrap();
        assert_eq!(
            (solution.value(x), solution.value(y), solution.value(z)),
            (2.0, 6.5, 1.0)
        )
    }
}
