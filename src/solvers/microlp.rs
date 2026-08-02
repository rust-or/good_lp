//! A solver that uses [microlp](https://docs.rs/microlp), a pure rust solver.

use std::time::Duration;

use microlp::{Error, SolveOptions, TerminationReason};

use crate::solvers::MipGapError;
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{Constraint, Variable, WithInitialSolution, WithMipGap};
use crate::{
    constraint::ConstraintReference,
    solvers::{
        ObjectiveDirection, ResolutionError, Solution, SolutionStatus, SolverModel, WithTimeLimit,
    },
};

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
    let mut initial_solution = Vec::with_capacity(variables.initial_solution_len());
    let mut construction_error = None;
    let variables: Vec<microlp::Variable> = variables
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
                let coeff = *objective.linear.coefficients.get(&var).unwrap_or(&0.);
                let microlp_var = if is_integer {
                    let min = if min == f64::NEG_INFINITY {
                        Some(i32::MIN)
                    } else {
                        let min = min.ceil();
                        (i32::MIN as f64..=i32::MAX as f64)
                            .contains(&min)
                            .then_some(min as i32)
                    };
                    let max = if max == f64::INFINITY {
                        Some(i32::MAX)
                    } else {
                        let max = max.floor();
                        (i32::MIN as f64..=i32::MAX as f64)
                            .contains(&max)
                            .then_some(max as i32)
                    };

                    match (min, max) {
                        (Some(min), Some(max)) if min <= max => {
                            problem.add_integer_var(coeff, (min, max))
                        }
                        (Some(_), Some(_)) => {
                            construction_error.get_or_insert(ResolutionError::Infeasible);
                            problem.add_integer_var(coeff, (0, 0))
                        }
                        _ => {
                            construction_error.get_or_insert_with(|| {
                                ResolutionError::Str(
                                    "Microlp only supports integer bounds within the i32 range"
                                        .to_string(),
                                )
                            });
                            problem.add_integer_var(coeff, (0, 0))
                        }
                    }
                } else {
                    problem.add_var(coeff, (min, max))
                };
                if let Some(value) = initial {
                    initial_solution.push((microlp_var, value));
                }
                microlp_var
            },
        )
        .collect();
    MicroLpProblem {
        problem,
        variables,
        n_constraints: 0,
        mip_gap: None,
        time_limit: None,
        initial_solution: (!initial_solution.is_empty()).then_some(initial_solution),
        construction_error,
    }
}

/// A microlp model
pub struct MicroLpProblem {
    problem: microlp::Problem,
    variables: Vec<microlp::Variable>,
    n_constraints: usize,
    mip_gap: Option<f32>,
    time_limit: Option<f64>,
    initial_solution: Option<Vec<(microlp::Variable, f64)>>,
    construction_error: Option<ResolutionError>,
}

impl MicroLpProblem {
    /// Get the inner microlp model
    pub fn as_inner(&self) -> &microlp::Problem {
        &self.problem
    }
}

impl WithTimeLimit for MicroLpProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.time_limit = Some(seconds.into());
        self
    }
}

impl SolverModel for MicroLpProblem {
    type Solution = MicroLpSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        if let Some(error) = self.construction_error {
            return Err(error);
        }
        let mut opts = SolveOptions::default();
        opts.mip_gap = self.mip_gap.unwrap_or(0.0) as f64;
        opts.time_limit = self.time_limit.map(Duration::from_secs_f64);
        opts.warm_start = self.initial_solution;
        // An interrupted outcome carries no variable values, so the time limit
        // expired before microlp found any feasible solution.
        let solution = self
            .problem
            .solve_with(opts)?
            .into_solution()
            .map_err(|_| {
                ResolutionError::Other("Time limit reached before finding a feasible solution")
            })?;
        Ok(MicroLpSolution {
            solution,
            variables: self.variables,
        })
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.n_constraints;
        let op = if constraint.is_equality() {
            microlp::ComparisonOp::Eq
        } else {
            microlp::ComparisonOp::Le
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

impl WithInitialSolution for MicroLpProblem {
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        self.initial_solution = Some(
            solution
                .into_iter()
                .map(|(variable, value)| (self.variables[variable.index()], value))
                .collect(),
        );
        self
    }
}

impl From<microlp::Error> for ResolutionError {
    fn from(microlp_error: Error) -> Self {
        match microlp_error {
            microlp::Error::Unbounded => Self::Unbounded,
            microlp::Error::Infeasible => Self::Infeasible,
            microlp::Error::InvalidOperation(s) => Self::Str(s),
            microlp::Error::InvalidOptions(s) => Self::Str(s),
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
        match self.solution.termination_reason() {
            // The optimality proof completed.
            TerminationReason::ProvenOptimal => SolutionStatus::Optimal,
            // A feasible incumbent was within the requested relative gap.
            TerminationReason::MipGap => SolutionStatus::GapLimit,
            // The time budget ran out with a feasible incumbent.
            TerminationReason::TimeLimit => SolutionStatus::TimeLimit,
            // The branch & bound node budget ran out with a feasible
            // incumbent, this is unused in good_lp
            TerminationReason::NodeLimit => SolutionStatus::TimeLimit,
            _ => SolutionStatus::TimeLimit,
        }
    }
    fn value(&self, variable: Variable) -> f64 {
        self.solution.var_value(self.variables[variable.index()])
    }
}

impl WithMipGap for MicroLpProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.mip_gap
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        if mip_gap.is_sign_negative() {
            return Err(MipGapError::Negative);
        } else if mip_gap.is_infinite() {
            return Err(MipGapError::Infinite);
        }

        self.mip_gap = Option::Some(mip_gap);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Expression, ResolutionError, Solution, SolverModel,
        solvers::{SolutionStatus, WithInitialSolution, WithMipGap, WithTimeLimit},
        variable, variables,
    };

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

    #[test]
    fn preserves_integer_points_inside_fractional_bounds() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().clamp(0.9, 2.1));

        let solution = vars.minimise(x).using(microlp).solve().unwrap();

        assert_eq!(solution.value(x), 1.0);
    }

    #[test]
    fn rejects_integer_bounds_without_an_integer_point() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().clamp(0.1, 0.9));

        assert!(matches!(
            vars.minimise(x).using(microlp).solve(),
            Err(ResolutionError::Infeasible)
        ));
    }

    #[test]
    fn rejects_integer_bounds_outside_microlps_domain() {
        let mut vars = variables!();
        let x = vars.add(
            variable()
                .integer()
                .clamp(i32::MAX as f64 + 1.0, i32::MAX as f64 + 2.0),
        );

        assert!(matches!(
            vars.minimise(x).using(microlp).solve(),
            Err(ResolutionError::Str(message)) if message.contains("i32 range")
        ));
    }

    #[test]
    fn uses_an_initial_mip_solution() {
        let mut vars = variables!();
        let a = vars.add(variable().integer().clamp(0, 10));
        let b = vars.add(variable().integer().clamp(0, 10));

        let solution = vars
            .minimise(3 * a + 4 * b)
            .using(microlp)
            .with((a + 2 * b) >> 5)
            .with((3 * a + b) >> 4)
            .with_initial_solution([(a, 3.0), (b, 1.0)])
            .with_mip_gap(0.5)
            .unwrap()
            .solve()
            .unwrap();

        assert!(matches!(solution.status(), SolutionStatus::GapLimit));
        assert_eq!((solution.value(a), solution.value(b)), (3.0, 1.0));
    }

    #[test]
    fn reports_a_time_limit_hit_before_any_solution() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        // A zero time limit stops microlp before it can solve the relaxation,
        // so there is no feasible solution to read values from.
        let result = vars
            .maximise(x + y)
            .using(microlp)
            .with((2 * x + y) << 4)
            .with_time_limit(0)
            .solve();
        assert!(matches!(result, Err(ResolutionError::Other(_))));
    }

    #[test]
    fn can_solve_with_gap_limit() {
        let (status_optimal, value_optimal) = knapsack_value(None);
        let (status_suboptimal, value_suboptimal) = knapsack_value(Some(0.5));

        assert!(matches!(status_optimal, SolutionStatus::Optimal));
        assert!(matches!(status_suboptimal, SolutionStatus::GapLimit));
        assert!(value_suboptimal <= value_optimal);
    }

    fn knapsack_value(mipgap: Option<f32>) -> (SolutionStatus, f64) {
        // (value, cost) of each object
        let objects: Vec<(f64, f64)> = vec![
            (1.87, 6.03),
            (3.22, 8.03),
            (9.91, 5.16),
            (8.31, 1.72),
            (7.00, 6.33),
            (5.15, 8.20),
            (8.01, 4.63),
            (2.22, 1.50),
            (7.04, 6.26),
            (8.99, 9.62),
            (2.13, 4.00),
            (8.02, 8.02),
            (3.07, 1.92),
            (1.98, 9.03),
            (7.23, 9.51),
            (4.08, 3.24),
            (9.65, 5.13),
            (6.53, 3.07),
            (6.76, 3.84),
            (9.63, 8.33),
        ];

        let mut prob_vars = variables!();
        let mut objective = Expression::with_capacity(objects.len());
        let mut constraint = Expression::with_capacity(objects.len());

        let budget: f64 = 25.0;
        for (value, cost) in objects {
            let var = prob_vars.add(variable().binary());
            objective.add_mul(value, var);
            constraint.add_mul(cost, var);
        }

        let mut model = prob_vars.maximise(objective.clone()).using(microlp);

        if let Some(gap) = mipgap {
            model = model.with_mip_gap(gap).unwrap();
        }

        model.add_constraint(constraint.leq(budget));

        let solution = model.solve().unwrap();

        // For this example we're interested only in the total value, not in the objects selected
        (solution.status(), objective.eval_with(&solution))
    }
}
