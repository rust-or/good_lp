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
    CardinalityConstraintSolver, WithInitialSolution,
};
use crate::{Constraint, Variable};

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
        let id = model.add_var(min, max, coeff, name.as_str(), var_type);
        var_map.insert(var, id);
    }

    SCIPProblem {
        model,
        id_for_var: var_map,
    }
}

/// The heuristic emphasis to use for the solver
pub enum ScipHeuristics {
    /// Use default values.
    Default,
    /// Set to aggressive settings.
    Aggressive,
    /// Set to fast settings.
    Fast,
    /// Turn off.
    Off,
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

    /// Sets whether or not SCIP should display verbose logging information to the console
    pub fn set_verbose(self, verbose: bool) -> Model<ProblemCreated> {
        self.model
            .set_int_param("display/verblevel", if verbose { 4 } else { 0 })
            .expect("could not set verbosity level")
    }

    /// Sets the heuristics parameter of the SCIP instance
    pub fn set_heuristics(self, heuristics: ScipHeuristics) -> Model<ProblemCreated> {
        self.model.set_heuristics(match heuristics {
            ScipHeuristics::Default => russcip::ParamSetting::Default,
            ScipHeuristics::Aggressive => russcip::ParamSetting::Aggressive,
            ScipHeuristics::Fast => russcip::ParamSetting::Fast,
            ScipHeuristics::Off => russcip::ParamSetting::Off,
        })
    }

    /// Sets the time limit in seconds
    pub fn set_time_limit(self, time_limit: usize) -> Model<ProblemCreated> {
        self.model.set_time_limit(time_limit)
    }

    /// Sets the memory limit in MB
    pub fn set_memory_limit(self, memory_limit: usize) -> Model<ProblemCreated> {
        self.model.set_memory_limit(memory_limit)
    }

    /// Sets a SCIP integer parameter
    pub fn set_option_int(self, option: &str, value: i32) -> Model<ProblemCreated> {
        self.model
            .set_int_param(option, value)
            .unwrap_or_else(|_| panic!("cound not set option {}", option))
    }

    /// Sets a SCIP long integer parameter
    pub fn set_option_longint(self, option: &str, value: i64) -> Model<ProblemCreated> {
        self.model
            .set_longint_param(option, value)
            .unwrap_or_else(|_| panic!("cound not set option {}", option))
    }

    /// Sets a SCIP boolean parameter
    pub fn set_option_bool(self, option: &str, value: bool) -> Model<ProblemCreated> {
        self.model
            .set_bool_param(option, value)
            .unwrap_or_else(|_| panic!("cound not set option {}", option))
    }

    /// Sets a SCIP string parameter
    pub fn set_option_str(self, option: &str, value: &str) -> Model<ProblemCreated> {
        self.model
            .set_str_param(option, value)
            .unwrap_or_else(|_| panic!("cound not set option {}", option))
    }

    /// Sets a SCIP real parameter
    pub fn set_option_float(self, option: &str, value: f64) -> Model<ProblemCreated> {
        self.model
            .set_real_param(option, value)
            .unwrap_or_else(|_| panic!("cound not set option {}", option))
    }
}

impl CardinalityConstraintSolver for SCIPProblem {
    /// Add cardinality constraint. Constrains the number of non-zero variables to at most `rhs`.
    fn add_cardinality_constraint(&mut self, vars: &[Variable], rhs: usize) -> ConstraintReference {
        let scip_vars = vars
            .iter()
            .map(|v| Rc::clone(&self.id_for_var[v]))
            .collect::<Vec<_>>();

        let index = self.model.n_conss() + 1;
        self.model
            .add_cons_cardinality(scip_vars, rhs, format!("cardinality{}", index).as_str());

        ConstraintReference { index }
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
            russcip::status::Status::Infeasible => Err(ResolutionError::Infeasible),
            russcip::status::Status::Unbounded => Err(ResolutionError::Unbounded),
            other_status => Err(ResolutionError::Str(format!(
                "Unexpected status {:?}",
                other_status
            ))),
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

    fn name() -> &'static str {
        "SCIP"
    }
}

impl WithInitialSolution for SCIPProblem {
    fn with_initial_solution(self, solution: impl IntoIterator<Item = (Variable, f64)>) -> Self {
        let sol = self.model.create_sol();
        for (var, val) in solution {
            sol.set_val(Rc::clone(&self.id_for_var[&var]), val);
        }
        self.model.add_sol(sol).expect("could not set solution");
        self
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
    use crate::{
        constraint, variable, variables, CardinalityConstraintSolver, Solution, SolverModel,
        WithInitialSolution,
    };

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
    fn can_solve_with_initial_solution() {
        // Solve problem initially
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(scip)
            .with((2 * x + y) << 4)
            .solve()
            .unwrap();
        // Recreate same problem with initial values slightly off
        let initial_x = solution.value(x) - 0.1;
        let initial_y = solution.value(x) - 1.0;
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2));
        let y = vars.add(variable().clamp(1, 3));
        let solution = vars
            .maximise(x + y)
            .using(scip)
            .with((2 * x + y) << 4)
            .with_initial_solution([(x, initial_x), (y, initial_y)])
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

    #[test]
    fn can_solve_cardinality_constraint() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2).integer());
        let y = vars.add(variable().clamp(0, 3).integer());
        let mut model = vars.maximise(5.0 * x + 3.0 * y).using(scip);
        model.add_cardinality_constraint(&[x, y], 1);
        let solution = model.solve().unwrap();
        assert_eq!((solution.value(x), solution.value(y)), (2., 0.));
    }
}
