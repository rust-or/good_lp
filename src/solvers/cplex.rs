//! A solver that uses the [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) solver.

use std::collections::HashMap;
use std::time::Duration;

use crate::{
    constraint::ConstraintReference, variable::UnsolvedProblem, Constraint, ResolutionError,
    Solution, SolutionStatus, SolverModel, Variable, VariableDefinition, WithTimeLimit,
};
use cplex_rs::parameters::TimeLimit;
use cplex_rs::{ConstraintType, Environment, Problem, ProblemType};

use super::ObjectiveDirection;

/// The [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) solver,
/// to be used with [UnsolvedProblem::using].
pub fn cplex(to_solve: UnsolvedProblem) -> CPLEXProblem {
    cplex_with_env(
        to_solve,
        Environment::new().expect("Unable to create cplex environment"),
    )
}

/// The [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) solver,
/// with an additional cplex_env parameter to specify the CPLEX enviroment
pub fn cplex_with_env(to_solve: UnsolvedProblem, cplex_env: Environment) -> CPLEXProblem {
    let mut model = Problem::new(cplex_env, "cplex problem")
        .expect("Unable to create cplex model")
        .set_objective_type(match to_solve.direction {
            ObjectiveDirection::Maximisation => cplex_rs::ObjectiveType::Maximize,
            ObjectiveDirection::Minimisation => cplex_rs::ObjectiveType::Minimize,
        })
        .expect("Unable to set cplex objective type");
    let mut id_for_var = HashMap::new();

    let mut has_integer = false;
    to_solve.variables.iter_variables_with_def().for_each(
        |(
            var,
            &VariableDefinition {
                min,
                max,
                is_integer,
                ref name,
                ..
            },
        )| {
            let coeff = *to_solve
                .objective
                .linear
                .coefficients
                .get(&var)
                .unwrap_or(&0.);
            let variable_type = if is_integer {
                has_integer = true;
                if min == 0. && max == 1. {
                    cplex_rs::VariableType::Binary
                } else {
                    cplex_rs::VariableType::Integer
                }
            } else {
                cplex_rs::VariableType::Continuous
            };
            let cplex_var = cplex_rs::Variable::new(variable_type, coeff, min, max, name);
            let cplex_id = model
                .add_variable(cplex_var)
                .expect("Unable to add cplex variable");

            id_for_var.insert(var, cplex_id);
        },
    );

    CPLEXProblem {
        model,
        has_integer,
        id_for_var,
    }
}

/// A CPLEX Model
pub struct CPLEXProblem {
    // the underlying CPLEX model representing the problem
    model: Problem,
    // whether the problem contains an integer variable or not
    has_integer: bool,
    // map from good_lp variables to SCIP variable ids
    id_for_var: HashMap<crate::Variable, cplex_rs::VariableId>,
}

impl CPLEXProblem {
    /// Get access to the raw cplex model
    pub fn as_inner(&self) -> &Problem {
        &self.model
    }

    /// Get mutable access to the raw cplex model
    pub fn as_inner_mut(&mut self) -> &mut Problem {
        &mut self.model
    }
}

impl SolverModel for CPLEXProblem {
    type Solution = CplexSolved;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let solution = match self.model.solve_as(if self.has_integer {
            ProblemType::MixedInteger
        } else {
            ProblemType::Linear
        }) {
            Ok(s) => s,
            Err(cplex_rs::Error::Cplex(cplex_rs::errors::Cplex::Unfeasible { .. })) => {
                return Err(ResolutionError::Infeasible);
            }
            Err(cplex_rs::Error::Cplex(cplex_rs::errors::Cplex::Unbounded { .. })) => {
                return Err(ResolutionError::Unbounded);
            }
            Err(cplex_rs::Error::Cplex(cplex_rs::errors::Cplex::Other { message, .. }))
            | Err(cplex_rs::Error::Input(cplex_rs::errors::Input { message })) => {
                return Err(ResolutionError::Str(message))
            }
        };

        Ok(CplexSolved {
            solution,
            id_for_var: self.id_for_var,
        })
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        let rhs = -c.expression.constant;
        let weighted_variables = c
            .expression
            .linear
            .coefficients
            .iter()
            .map(|(var, &coeff)| (self.id_for_var[var], coeff))
            .collect::<Vec<_>>();
        let con_type = if c.is_equality {
            ConstraintType::Eq
        } else {
            ConstraintType::LessThanEq
        };

        let cplex_con = cplex_rs::Constraint::new(con_type, rhs, c.name, weighted_variables);
        let con_index = self
            .model
            .add_constraint(cplex_con)
            .expect("Unable to add constraint to cplex model, aborting");

        ConstraintReference {
            index: con_index.into_inner(),
        }
    }

    fn name() -> &'static str {
        "CPLEX"
    }
}

impl WithTimeLimit for CPLEXProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.model
            .env_mut()
            .set_parameter(TimeLimit(Duration::from_secs_f64(seconds.into())))
            .expect("Failed to set CPLEX time limit");
        self
    }
}

/// A wrapper to a solved SCIP problem
pub struct CplexSolved {
    solution: cplex_rs::Solution,
    id_for_var: HashMap<crate::Variable, cplex_rs::VariableId>,
}

impl Solution for CplexSolved {
    fn status(&self) -> SolutionStatus {
        SolutionStatus::Optimal
    }
    fn value(&self, var: Variable) -> f64 {
        let id = &self.id_for_var[&var];
        self.solution.variable_values()[id.into_inner()]
    }
}

#[cfg(test)]
mod tests {
    use cplex_rs::Environment;

    use crate::{constraint, variable, variables, Solution, SolverModel};

    use super::cplex_with_env;

    #[test]
    fn can_solve_with_inequality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(|to_solve| {
                let cplex_env = Environment::new().unwrap();
                cplex_with_env(to_solve, cplex_env)
            })
            .with((2 * x + y) << 4)
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (0.5, 3.))
    }

    #[test]
    fn can_solve_with_equality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2).integer());
        let y = vars.add(variable().clamp(1, 3).integer());
        let solution = vars
            .maximise(x + y)
            .using(|to_solve| {
                let cplex_env = Environment::new().unwrap();
                cplex_with_env(to_solve, cplex_env)
            })
            .with(constraint!(2 * x + y == 4))
            .with(constraint!(x + 2 * y <= 5))
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (1., 2.));
    }
}
