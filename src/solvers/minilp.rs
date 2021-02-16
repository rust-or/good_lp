//! A solver that uses [minilp](https://docs.rs/minilp), a pure rust solver.

use crate::solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel};
use crate::variable::{UnsolvedProblem, VariableDefinition};
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
    let variables: Vec<minilp::Variable> = variables
        .iter_variables_with_def()
        .map(|(var, &VariableDefinition { min, max, .. })| {
            let coeff = *objective.linear.coefficients.get(&var).unwrap_or(&0.);
            problem.add_var(coeff, (min, max))
        })
        .collect();
    MiniLpProblem { problem, variables }
}

/// A minilp model
pub struct MiniLpProblem {
    problem: minilp::Problem,
    variables: Vec<minilp::Variable>,
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

    fn with(mut self, constraint: Constraint) -> Self {
        let coefficients: Vec<(minilp::Variable, f64)> = constraint
            .expression
            .linear
            .coefficients
            .iter()
            .map(|(var, &coeff)| (self.variables[var.index()], coeff))
            .collect();
        let op = match constraint.is_equality {
            true => minilp::ComparisonOp::Eq,
            false => minilp::ComparisonOp::Le,
        };
        let constant = -constraint.expression.constant;
        self.problem.add_constraint(&coefficients, op, constant);
        self
    }

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        match self.problem.solve() {
            Err(minilp::Error::Unbounded) => Err(ResolutionError::Unbounded),
            Err(minilp::Error::Infeasible) => Err(ResolutionError::Infeasible),
            Ok(solution) => Ok(MiniLpSolution {
                solution,
                variables: self.variables,
            }),
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
    use super::minilp;
    use crate::{variable, variables, Solution, SolverModel};

    #[test]
    fn can_solve_easy() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(minilp)
            .with(2 * x + y << 4)
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (0.5, 3.))
    }
}
