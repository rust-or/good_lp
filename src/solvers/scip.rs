//! A solver that uses [SCIP](https://scipopt.org), one
//! of the fastest non-commercial solvers for mixed integer programming.

use std::collections::HashMap;

use russcip::model::Model;
use russcip::model::ObjSense;
use russcip::variable::VarType;

use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};
use crate::variable::{UnsolvedProblem, VariableDefinition};

/// The [SCIP](https://scipopt.org) solver,
/// to be used with [UnsolvedProblem::using].
pub fn scip(to_solve: UnsolvedProblem) -> SCIPProblem {
    let mut problem = Model::new();

    problem.include_default_plugins();
    problem.create_prob("problem");
    problem.set_obj_sense(match to_solve.direction {
        ObjectiveDirection::Maximisation => ObjSense::Maximize,
        ObjectiveDirection::Minimisation => ObjSense::Minimize,
    });

    for (var, &VariableDefinition {
        min,
        max,
        is_integer,
        ref name
    }) in to_solve.variables.iter_variables_with_def() {
        let coeff = *to_solve.objective.linear.coefficients.get(&var).unwrap_or(&0.);
        let var_type = match is_integer {
            true => { VarType::Integer }
            false => { VarType::Continuous }
        };
        problem.add_var(min, max, coeff, name.as_str().clone(), var_type);
    }

    SCIPProblem {
        problem
    }
}

/// A SCIP Model
pub struct SCIPProblem {
    problem: Model,
}

impl SolverModel for SCIPProblem {
    type Solution = SCIPSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let vars = self.problem.get_vars();
        self.problem.solve();
        let sol = self.problem.get_best_sol();
        let values = vars.iter().map(|var| {
            (var.get_index(), sol.get_var_val(var))
        }).collect();

        let status = self.problem.get_status();
        match status {
            russcip::status::Status::OPTIMAL => {
                Ok(SCIPSolution {
                    values,
                })
            }
            russcip::status::Status::INFEASIBLE => {
                return Err(ResolutionError::Infeasible);
            }
            russcip::status::Status::UNBOUNDED => {
                return Err(ResolutionError::Unbounded);
            }
            other_status => {
                return Err(ResolutionError::Str(format!("Unexpected status {:?}", other_status)));
            }
        }
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        let constant = -c.expression.constant;
        let lhs = match c.is_equality {
            true => constant,
            false => -f64::INFINITY,
        };

        let mut vars = HashMap::new();
        for var in self.problem.get_vars() {
            let var_index = var.get_index();
            vars.insert(var_index, var);
        }

        let mut vars_in_cons = vec![];
        let mut coeffs = vec![];

        for (&var, &coeff) in c.expression.linear.coefficients.iter() {
            vars_in_cons.push(&vars[&var.index()]);
            coeffs.push(coeff);
        }

        let index = self.problem.get_n_conss() + 1;
        self.problem.add_cons(&vars_in_cons, &coeffs, lhs, constant, format!("c{}", index).as_str());

        ConstraintReference { index }
    }
}

/// A solution to a SCIP problem
pub struct SCIPSolution {
    values: HashMap<usize, f64>,
}

impl Solution for SCIPSolution {
    fn value(&self, var: Variable) -> f64 {
        self.values[&var.index()]
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint, Solution, SolverModel, variable, variables};

    use super::scip;

    #[test]
    fn can_solve_with_inequality() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(scip)
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
            .using(scip)
            .with(constraint!(2*x + y == 4))
            .with(constraint!(x + 2*y <= 5))
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (1., 2.));
    }
}
