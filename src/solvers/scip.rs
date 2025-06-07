//! A solver that uses [SCIP](https://scipopt.org), one
//! of the fastest non-commercial solvers for mixed integer programming.

use std::collections::HashMap;

use russcip::model::Model;
use russcip::model::ModelWithProblem;
use russcip::model::ObjSense;
use russcip::model::ProblemCreated;
use russcip::model::Solved;
use russcip::variable::VarType;
use russcip::ProblemOrSolving;
use russcip::Retcode;
use russcip::WithSolutions;

use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{
        MipGapError, ObjectiveDirection, ResolutionError, Solution, SolutionStatus, SolverModel,
    },
    CardinalityConstraintSolver, WithInitialSolution, WithMipGap, WithTimeLimit,
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
    let mut initial_solution = Vec::with_capacity(to_solve.variables.initial_solution_len());

    for (
        var,
        &VariableDefinition {
            min,
            max,
            initial,
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
        if let Some(val) = initial {
            initial_solution.push((var, val));
        };
    }

    let mut problem = SCIPProblem {
        model,
        id_for_var: var_map,
    };
    if !initial_solution.is_empty() {
        problem = problem.with_initial_solution(initial_solution);
    }
    problem
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
    id_for_var: HashMap<Variable, russcip::Variable>,
}

/// A value that can be read for a SCIP option
pub trait ScipOptionGetValue {
    /// Obtains the value of a given model for a given option name
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self
    where
        Self: Sized;
}

/// A value that can be set for a SCIP option
pub trait ScipOptionSetValue {
    /// Applies the value to a given model for a given option name
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode>
    where
        Self: Sized;
}

impl ScipOptionGetValue for i32 {
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self {
        model.int_param(option)
    }
}
impl ScipOptionSetValue for i32 {
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode> {
        model.set_int_param(option, self)
    }
}

impl ScipOptionGetValue for f64 {
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self {
        model.real_param(option)
    }
}
impl ScipOptionSetValue for f64 {
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode> {
        model.set_real_param(option, self)
    }
}
impl ScipOptionGetValue for String {
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self {
        model.str_param(option)
    }
}
impl ScipOptionSetValue for &str {
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode> {
        model.set_str_param(option, &self)
    }
}
impl ScipOptionGetValue for i64 {
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self {
        model.longint_param(option)
    }
}
impl ScipOptionSetValue for i64 {
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode> {
        model.set_longint_param(option, self)
    }
}
impl ScipOptionGetValue for bool {
    fn get_for(model: Model<ProblemCreated>, option: &str) -> Self {
        model.bool_param(option)
    }
}
impl ScipOptionSetValue for bool {
    fn set_for(
        self,
        model: Model<ProblemCreated>,
        option: &str,
    ) -> Result<Model<ProblemCreated>, Retcode> {
        model.set_bool_param(option, self)
    }
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
    pub fn try_set_verbose(mut self, verbose: bool) -> Result<Self, Retcode> {
        self.model = self
            .model
            .set_int_param("display/verblevel", if verbose { 4 } else { 0 })?;
        Ok(self)
    }

    /// Tries to set whether or not SCIP should display verbose logging
    /// information to the console and panics if the operation fails
    pub fn set_verbose(self, verbose: bool) -> Self {
        self.try_set_verbose(verbose)
            .unwrap_or_else(|e| panic!("cound not set verbosity to {}: {:?}", verbose, e))
    }

    /// Sets the heuristics parameter of the SCIP instance
    pub fn set_heuristics(mut self, heuristics: ScipHeuristics) -> Self {
        self.model = self.model.set_heuristics(match heuristics {
            ScipHeuristics::Default => russcip::ParamSetting::Default,
            ScipHeuristics::Aggressive => russcip::ParamSetting::Aggressive,
            ScipHeuristics::Fast => russcip::ParamSetting::Fast,
            ScipHeuristics::Off => russcip::ParamSetting::Off,
        });
        self
    }

    /// Sets the time limit in seconds
    pub fn set_time_limit(mut self, time_limit: usize) -> Self {
        self.model = self.model.set_time_limit(time_limit);
        self
    }

    /// Sets the memory limit in MB
    pub fn set_memory_limit(mut self, memory_limit: usize) -> Self {
        self.model = self.model.set_memory_limit(memory_limit);
        self
    }

    /// Sets a SCIP parameter
    pub fn try_set_option<T: ScipOptionSetValue>(
        mut self,
        option: &str,
        value: T,
    ) -> Result<Self, Retcode> {
        self.model = value.set_for(self.model, option)?;
        Ok(self)
    }

    /// Tries to set a SCIP parameter and panics if the operation fails
    pub fn set_option<T: ScipOptionSetValue>(self, option: &str, value: T) -> Self {
        self.try_set_option(option, value)
            .unwrap_or_else(|e| panic!("could not set option '{}': {:?}", option, e))
    }

    /// Reads a SCIP parameter
    pub fn get_option<T: ScipOptionGetValue>(self, option: &str) -> T {
        T::get_for(self.model, option)
    }

    /// Set an initial solution for the problem.
    /// Returns an error if the solution is not feasible.
    fn try_with_initial_solution(
        self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Result<Self, russcip::SolError> {
        let sol = self.model.create_orig_sol();
        for (var, val) in solution {
            sol.set_val(&self.id_for_var[&var], val);
        }
        self.model.add_sol(sol)?;
        Ok(self)
    }
}

impl CardinalityConstraintSolver for SCIPProblem {
    /// Add cardinality constraint. Constrains the number of non-zero variables to at most `rhs`.
    fn add_cardinality_constraint(&mut self, vars: &[Variable], rhs: usize) -> ConstraintReference {
        let Self { id_for_var, model } = self;
        let scip_vars: Vec<&russcip::Variable> = vars.iter().map(|v| &id_for_var[v]).collect();
        let index = model.n_conss() + 1;
        model.add_cons_cardinality(scip_vars, rhs, format!("cardinality{}", index).as_str());
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
            russcip::Status::TimeLimit => Ok(SCIPSolved {
                status: SolutionStatus::TimeLimit,
                solved_problem: solved_model,
                id_for_var: self.id_for_var,
            }),
            russcip::Status::GapLimit => Ok(SCIPSolved {
                status: SolutionStatus::GapLimit,
                solved_problem: solved_model,
                id_for_var: self.id_for_var,
            }),
            russcip::status::Status::Optimal => Ok(SCIPSolved {
                status: SolutionStatus::Optimal,
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
            let id = &self.id_for_var[&var];
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
    /// Set an initial solution for the problem.
    ///
    /// # Panics
    ///
    /// Panics if the solution is not feasible.
    fn with_initial_solution(self, solution: impl IntoIterator<Item = (Variable, f64)>) -> Self {
        self.try_with_initial_solution(solution)
            .expect("could not set solution")
    }
}
impl WithMipGap for SCIPProblem {
    fn mip_gap(&self) -> Option<f32> {
        Some(self.model.real_param("limits/gap") as f32)
    }
    fn with_mip_gap(self, mip_gap: f32) -> Result<Self, MipGapError>
    where
        Self: Sized,
    {
        self.try_set_option::<f64>("limits/gap", mip_gap.into())
            .map_err(|e| {
                if mip_gap.is_sign_negative() {
                    MipGapError::Negative
                } else if mip_gap.is_infinite() {
                    MipGapError::Infinite
                } else {
                    MipGapError::Other(format!("cannot set mip gap error: {:?}", e))
                }
            })
    }
}

impl WithTimeLimit for SCIPProblem {
    fn with_time_limit<T: Into<f64>>(self, seconds: T) -> Self {
        self.set_time_limit(seconds.into() as usize)
    }
}

/// A wrapper to a solved SCIP problem
pub struct SCIPSolved {
    status: SolutionStatus,
    solved_problem: Model<Solved>,
    id_for_var: HashMap<Variable, russcip::Variable>,
}

impl Solution for SCIPSolved {
    fn status(&self) -> SolutionStatus {
        self.status
    }
    fn value(&self, var: Variable) -> f64 {
        self.solved_problem
            .best_sol()
            .expect("This problem is expected to have Optimal status, a ")
            .val(&self.id_for_var[&var])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        constraint, variable, variables, CardinalityConstraintSolver, Expression, Solution,
        SolutionStatus, SolverModel, WithInitialSolution, WithMipGap,
    };

    use super::scip;

    #[test]
    fn can_solve_with_gap_limit() {
        let (status_optimal, value_optimal) = knapsack_value(None);
        let (status_suboptimal, value_suboptimal) = knapsack_value(Some(0.5));

        assert!(matches!(status_optimal, SolutionStatus::Optimal));
        assert!(matches!(status_suboptimal, SolutionStatus::GapLimit));
        assert!(value_suboptimal < value_optimal);
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

        let mut model = prob_vars.maximise(objective.clone()).using(scip);

        if let Some(gap) = mipgap {
            model = model.with_mip_gap(gap).unwrap();
        }

        model.add_constraint(constraint.leq(budget));

        let solution = model.solve().unwrap();

        // For this example we're interested only in the total value, not in the objects selected
        (solution.status(), objective.eval_with(&solution))
    }

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
        let initial_y = solution.value(y) - 1.0;
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
    fn solve_problem_with_initial_variable_values() {
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
        let initial_y = solution.value(y) - 1.0;
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 2).initial(initial_x));
        let y = vars.add(variable().clamp(1, 3).initial(initial_y));
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
            .set_verbose(true)
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

    #[test]
    fn can_set_int_param() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 1));
        let model = vars
            .maximise(x)
            .using(scip)
            .set_option("display/verblevel", 2);
        assert_eq!(model.get_option::<i32>("display/verblevel"), 2);
    }

    #[test]
    fn can_set_real_param() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 1));
        let model = vars.maximise(x).using(scip).set_option("limits/time", 0.);
        assert_eq!(model.get_option::<f64>("limits/time"), 0.);
    }

    #[test]
    fn can_set_str_param() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 1));
        let model = vars
            .maximise(x)
            .using(scip)
            .set_option("concurrent/paramsetprefix", "custom/path");
        assert_eq!(
            &model.get_option::<String>("concurrent/paramsetprefix"),
            "custom/path"
        );
    }

    #[test]
    fn can_set_longint_param() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 1));
        let model = vars
            .maximise(x)
            .using(scip)
            .set_option::<i64>("constraints/components/nodelimit", 99);
        assert_eq!(
            model.get_option::<i64>("constraints/components/nodelimit"),
            99
        );
    }

    #[test]
    fn can_set_bool_param() {
        let mut vars = variables!();
        let x = vars.add(variable().clamp(0, 1));
        let model = vars
            .maximise(x)
            .using(scip)
            .set_option("display/allviols", true);
        assert_eq!(model.get_option::<bool>("display/allviols"), true);
    }
}
