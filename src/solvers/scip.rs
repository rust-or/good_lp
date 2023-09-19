//! A solver that uses [SCIP](https://scipopt.org), one
//! of the fastest non-commercial solvers for mixed integer programming.

use std::collections::HashMap;
use std::rc::Rc;

use russcip::model::Model;
use russcip::model::ModelWithProblem;
use russcip::model::ObjSense;
use russcip::model::ProblemCreated;
use russcip::model::Solved;
use russcip::variable::VarType;
use russcip::ProblemOrSolving;
use russcip::WithSolutions;

use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};
use crate::{Constraint, Variable};

/// String constant storing the name of the default solver.
/// Re-exported in `src/lib.rs`.
pub const DEFAULT_SOLVER_NAME: &str = "scip";

/// The [SCIP](https://scipopt.org) solver,
/// to be used with [UnsolvedProblem::using].
pub fn scip(to_solve: UnsolvedProblem) -> SCIPProblem {
    let mut model = Model::new()
        .hide_output()
        .include_default_plugins()
        .create_prob("problem")
        .set_obj_sense(match to_solve.direction {
            ObjectiveDirection::Maximisation => ObjSense::Maximize,
            ObjectiveDirection::Minimisation => ObjSense::Minimize,
        });
    let mut var_map = HashMap::new();

    for (
        var,
        &VariableDefinition {
            min,
            max,
            is_integer,
            ref name,
        },
    ) in to_solve.variables.iter_variables_with_def()
    {
        let coeff = *to_solve
            .objective
            .linear
            .coefficients
            .get(&var)
            .unwrap_or(&0.);
        let var_type = match is_integer {
            true => VarType::Integer,
            false => VarType::Continuous,
        };
        let id = model.add_var(min, max, coeff, name.as_str().clone(), var_type);
        var_map.insert(var, id);
    }

    SCIPProblem {
        model: model,
        id_for_var: var_map,
    }
}

/// A SCIP Model
pub struct SCIPProblem {
    // the underlying SCIP model representing the problem
    model: Model<ProblemCreated>,
    // map from good_lp variables to SCIP variable ids
    id_for_var: HashMap<Variable, Rc<russcip::Variable>>,
}

impl SCIPProblem {
    /// Get access to the raw russcip model
    pub fn as_inner(&self) -> &Model<ProblemCreated> {
        &self.model
    }

    /// Get mutable access to the raw russcip model
    pub fn as_inner_mut(&mut self) -> &mut Model<ProblemCreated> {
        &mut self.model
    }
}

impl SolverModel for SCIPProblem {
    type Solution = SCIPSolved;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let solved_model = self.model.solve();
        let status = solved_model.status();
        match status {
            russcip::status::Status::Optimal => Ok(SCIPSolved {
                solved_problem: solved_model,
                id_for_var: self.id_for_var,
            }),
            russcip::status::Status::Infeasible => {
                return Err(ResolutionError::Infeasible);
            }
            russcip::status::Status::Unbounded => {
                return Err(ResolutionError::Unbounded);
            }
            other_status => {
                return Err(ResolutionError::Str(format!(
                    "Unexpected status {:?}",
                    other_status
                )));
            }
        }
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        let constant = -c.expression.constant;
        let lhs = match c.is_equality {
            true => constant,
            false => -f64::INFINITY,
        };

        let n_vars_in_cons = c.expression.linear.coefficients.len();
        let mut vars_in_cons = Vec::with_capacity(n_vars_in_cons);
        let mut coeffs = Vec::with_capacity(n_vars_in_cons);
        for (&var, &coeff) in c.expression.linear.coefficients.iter() {
            let id = Rc::clone(&self.id_for_var[&var]);
            vars_in_cons.push(id);
            coeffs.push(coeff);
        }

        let index = self.model.n_conss() + 1;
        self.model.add_cons(
            vars_in_cons,
            &coeffs,
            lhs,
            constant,
            format!("c{}", index).as_str(),
        );

        ConstraintReference { index }
    }
}

/// A wrapper to a solved SCIP problem
pub struct SCIPSolved {
    solved_problem: Model<Solved>,
    id_for_var: HashMap<Variable, Rc<russcip::Variable>>,
}

impl Solution for SCIPSolved {
    fn value(&self, var: Variable) -> f64 {
        let sol = self
            .solved_problem
            .best_sol()
            .expect("This problem is expected to have Optimal status, a ");
        let id = Rc::clone(&self.id_for_var[&var]);
        sol.val(id)
    }
}

#[cfg(test)]
mod tests {
    use crate::{constraint, variable, variables, Solution, SolverModel};

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
            .with(constraint!(2 * x + y == 4))
            .with(constraint!(x + 2 * y <= 5))
            .solve()
            .unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (1., 2.));
    }
}
