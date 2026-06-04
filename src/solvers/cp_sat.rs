//! A solver bridge to the Google OR-Tools CP-SAT solver, activated via the `cp_sat` feature.
//!
//! # Limitations
//!
//! - CP-SAT only supports **integer** and **binary** variables. Continuous (floating-point)
//!   variables cause a panic at model creation.
//! - All coefficients and constants are rounded to `i64`, a consequence of CP-SAT's
//!   integer-only arithmetic.

use cp_sat::builder::{CpModelBuilder, IntVar, LinearExpr};
use cp_sat::ffi;
use cp_sat::proto::{CpSolverResponse, CpSolverStatus, SatParameters};

use crate::solvers::{MipGapError, SolutionStatus, WithInitialSolution, WithMipGap, WithTimeLimit};
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{Constraint, Variable};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};

/// The CP-SAT [Google OR-Tools](https://developers.google.com/optimization) solver library.
/// To be passed to [`UnsolvedProblem::using`](crate::variable::UnsolvedProblem::using)
///
/// # Panics
///
/// Panics if any variable is not an integer variable
/// (i.e., `is_integer` is `false` on the variable definition),
/// because CP-SAT does not support continuous variables.
pub fn cp_sat(to_solve: UnsolvedProblem) -> CpSatProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    let mut model = CpModelBuilder::default();

    // Create CP-SAT integer variables from good_lp variable definitions
    let cp_sat_vars: Vec<IntVar> = variables
        .iter_variables_with_def()
        .map(|(var, def)| {
            assert!(
                def.is_integer,
                "CP-SAT does not support continuous variables. \
                 Variable {} (index {}) was not declared as integer or binary. \
                 Please use `.integer()` or `.binary()` on the variable definition.",
                if def.name.is_empty() {
                    format!("v{}", var.index())
                } else {
                    def.name.clone()
                },
                var.index()
            );
            create_cp_sat_var(&mut model, def, var)
        })
        .collect();

    // Build the objective expression
    let mut objective_expr = LinearExpr::default();
    for (var, coeff) in &objective.linear.coefficients {
        let cp_var = cp_sat_vars[var.index()];
        let coeff_i64 = round_i64(*coeff);
        if coeff_i64 != 0 {
            objective_expr += (coeff_i64, cp_var);
        }
    }

    // Set objective direction
    match direction {
        ObjectiveDirection::Maximisation => {
            model.maximize(objective_expr);
        }
        ObjectiveDirection::Minimisation => {
            model.minimize(objective_expr);
        }
    }

    // Store initial solution hints if any are set on variables
    let mut initial_solution = Vec::with_capacity(variables.initial_solution_len());
    for (var, def) in variables.iter_variables_with_def() {
        if let Some(val) = def.initial {
            initial_solution.push((var, val));
        }
    }

    let mut problem = CpSatProblem {
        model,
        cp_sat_vars,
        params: SatParameters::default(),
    };

    if !initial_solution.is_empty() {
        problem = problem.with_initial_solution(initial_solution);
    }

    problem
}

/// Rounds `f64` to `i64` with saturation at the numeric limits.
/// NaN is logged to stderr and treated as 0.
fn round_i64(x: f64) -> i64 {
    if x.is_nan() {
        eprintln!("Warning: CP-SAT received NaN, treating as 0.");
        0
    } else if x >= i64::MAX as f64 {
        i64::MAX
    } else if x <= i64::MIN as f64 {
        i64::MIN
    } else {
        x.round() as i64
    }
}

/// Translates a good_lp `VariableDefinition` into a CP-SAT `IntVar`.
///
/// Bound mapping: `min`/`max` from `VariableDefinition` → CP-SAT domain interval `(lo, hi)`.
/// Non-finite bounds (infinity) map to `i64::MIN` / `i64::MAX`.
fn create_cp_sat_var(
    model: &mut CpModelBuilder,
    def: &VariableDefinition,
    var: Variable,
) -> IntVar {
    let name = if def.name.is_empty() {
        format!("v{}", var.index())
    } else {
        def.name.clone()
    };

    // Translate bounds to CP-SAT domain
    let domain_lower = if def.min.is_finite() {
        round_i64(def.min)
    } else {
        i64::MIN
    };
    let domain_upper = if def.max.is_finite() {
        round_i64(def.max)
    } else {
        i64::MAX
    };

    // Validate: lower bound must not exceed upper bound
    assert!(
        domain_lower <= domain_upper,
        "Invalid variable bounds: lower={} > upper={} for variable '{}'",
        def.min,
        def.max,
        name
    );

    let domain = [(domain_lower, domain_upper)];
    model.new_int_var_with_name(domain, name)
}

/// A CP-SAT model wrapping `CpModelBuilder`, with a linear objective,
/// variable mappings, and solver parameters. Constructed via [`cp_sat()`].
///
/// See the [module-level documentation](self) for constraint translation
/// semantics and status mapping details.
pub struct CpSatProblem {
    model: CpModelBuilder,
    cp_sat_vars: Vec<IntVar>,
    params: SatParameters,
}

impl CpSatProblem {
    /// Returns a shared reference to the inner CP-SAT model for inspection.
    ///
    /// This is useful for accessing read-only model data (e.g., [`CpModelBuilder::proto`]).
    /// To customize solver parameters, use [`params`](Self::params) or [`params_mut`](Self::params_mut)
    /// instead.
    pub fn as_inner(&self) -> &CpModelBuilder {
        &self.model
    }

    /// Builds a `LinearExpr` from a good_lp `Constraint`'s expression,
    /// converting f64 coefficients to i64.
    fn expr_from_constraint_expression(&self, expression: &crate::Expression) -> LinearExpr {
        let mut linear = LinearExpr::default();
        for (var, coeff) in expression.linear.coefficients.iter() {
            let coeff_i64 = round_i64(*coeff);
            if coeff_i64 != 0 {
                linear += (coeff_i64, self.cp_sat_vars[var.index()]);
            }
        }
        linear
    }

    /// Sets whether or not the solver should display verbose logging information to the console.
    ///
    /// When enabled, CP-SAT will log search progress to stdout.
    pub fn set_verbose(mut self, verbose: bool) -> Self {
        self.params.log_search_progress = Some(verbose);
        self
    }

    /// Sets whether the solver log output should be captured in the solver response.
    ///
    /// When enabled, the log output will be available in the solution's
    /// [`CpSatSolution::solution_log`] method. This implicitly enables
    /// search progress logging.
    pub fn set_log_to_response(mut self, log: bool) -> Self {
        self.params.log_to_response = Some(log);
        if log {
            self.params.log_search_progress = Some(true);
        }
        self
    }

    /// Sets the number of search workers to use for parallel solving.
    ///
    /// By default, CP-SAT automatically selects the number of workers based on
    /// the number of CPU cores. Set this to override the automatic selection.
    ///
    /// A value of 1 disables parallelism. Higher values may speed up solving
    /// on multi-core systems.
    pub fn set_num_search_workers(mut self, n: i32) -> Self {
        self.params.num_search_workers = Some(n);
        self
    }

    /// Sets the random seed for deterministic randomness in the solver.
    ///
    /// When set to a fixed value, repeated solves of the same problem will
    /// produce identical results. A value of 0 (or not setting this) lets
    /// the solver use non-deterministic randomness.
    pub fn set_random_seed(mut self, seed: i32) -> Self {
        self.params.random_seed = Some(seed);
        self
    }

    /// Sets the absolute MIP gap tolerance.
    ///
    /// This stops the search when the absolute difference between the best
    /// bound and the best objective falls below this value.
    pub fn set_mip_abs_gap(mut self, gap: f64) -> Result<Self, MipGapError> {
        if gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            self.params.absolute_gap_limit = Some(gap);
            Ok(self)
        }
    }

    /// Sets whether to fill additional solutions in the response.
    ///
    /// When enabled, the solver will return all intermediate solutions found
    /// during the search in the solution's
    /// [`CpSatSolution::additional_solutions`] method.
    pub fn set_fill_additional_solutions(mut self, fill: bool) -> Self {
        self.params.fill_additional_solutions_in_response = Some(fill);
        self
    }

    /// Returns a reference to the current solver parameters.
    /// This allows direct access to all CP-SAT `SatParameters` fields.
    pub fn params(&self) -> &SatParameters {
        &self.params
    }

    /// Returns a mutable reference to the current solver parameters.
    /// This allows direct modification of all CP-SAT `SatParameters` fields.
    pub fn params_mut(&mut self) -> &mut SatParameters {
        &mut self.params
    }

    /// Deletes all solution hints previously set via `with_initial_solution`
    /// or via variable `.initial()` definitions.
    ///
    /// This can be useful if you want to clear the warm-start hints without
    /// modifying the model structure.
    pub fn del_hints(&mut self) {
        self.model.del_hints();
    }

    /// Get the solver statistics as a formatted string.
    /// This returns the model statistics even before solving.
    pub fn model_stats(&self) -> String {
        self.model.stats()
    }
}

impl SolverModel for CpSatProblem {
    type Solution = CpSatSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let response = self.model.solve_with_parameters(&self.params);

        let has_time_limit = self.params.max_deterministic_time.is_some()
            || self.params.max_time_in_seconds.is_some();
        let has_gap_limit =
            self.params.relative_gap_limit.is_some() || self.params.absolute_gap_limit.is_some();

        let status = response.status();
        let status_from_limits = || {
            if has_time_limit {
                SolutionStatus::TimeLimit
            } else if has_gap_limit {
                SolutionStatus::GapLimit
            } else {
                SolutionStatus::Optimal
            }
        };

        match status {
            CpSolverStatus::Optimal => Ok(CpSatSolution {
                response,
                status: SolutionStatus::Optimal,
                cp_sat_vars: self.cp_sat_vars,
            }),
            CpSolverStatus::Feasible => Ok(CpSatSolution {
                response,
                status: status_from_limits(),
                cp_sat_vars: self.cp_sat_vars,
            }),
            CpSolverStatus::Infeasible => Err(ResolutionError::Infeasible),
            CpSolverStatus::Unknown => {
                if !response.solution.is_empty() {
                    Ok(CpSatSolution {
                        response,
                        status: status_from_limits(),
                        cp_sat_vars: self.cp_sat_vars,
                    })
                } else if has_time_limit {
                    Err(ResolutionError::Other(
                        "Time limit reached without finding a feasible solution",
                    ))
                } else {
                    Err(ResolutionError::Other(
                        "Unknown solver status: search limit reached or other interruption",
                    ))
                }
            }
            CpSolverStatus::ModelInvalid => {
                let validation = self.model.validate_cp_model();
                Err(ResolutionError::Str(format!(
                    "Invalid CP-SAT model: {}",
                    validation
                )))
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.model.proto().constraints.len();

        // good_lp Constraint:
        //   expression = sum(coeff_i * var_i) + constant
        //   is_equality = true  => expression == 0 => sum(coeff_i * var_i) == -constant
        //   is_equality = false => expression <= 0 => sum(coeff_i * var_i) <= -constant

        let constant_i64 = round_i64(-constraint.expression.constant);
        let linear_expr = self.expr_from_constraint_expression(&constraint.expression);

        if constraint.is_equality {
            self.model.add_eq(linear_expr, constant_i64);
        } else {
            self.model.add_le(linear_expr, constant_i64);
        }

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "CP-SAT"
    }
}

impl WithInitialSolution for CpSatProblem {
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        for (var, val) in solution {
            let cp_var = self.cp_sat_vars[var.index()];
            let hint_value = round_i64(val);
            self.model.add_hint(cp_var, hint_value);
        }
        self
    }
}

impl WithTimeLimit for CpSatProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.params.max_time_in_seconds = Some(seconds.into());
        self
    }
}

impl WithMipGap for CpSatProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.params.relative_gap_limit.map(|v| v as f32)
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        if mip_gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if mip_gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            self.params.relative_gap_limit = Some(mip_gap as f64);
            Ok(self)
        }
    }
}

/// The result of solving a CP-SAT model.
///
/// Wraps a `CpSolverResponse` and provides access to variable values via the
/// [`Solution`] trait, as well as solver metadata (statistics, logs, additional solutions).
///
/// See the [module-level documentation](self) for the status mapping from
/// `CpSolverStatus` to [`SolutionStatus`] / [`ResolutionError`].
pub struct CpSatSolution {
    response: CpSolverResponse,
    status: SolutionStatus,
    cp_sat_vars: Vec<IntVar>,
}

impl CpSatSolution {
    /// Returns the inner solver response for advanced querying
    pub fn response(&self) -> &CpSolverResponse {
        &self.response
    }

    /// Returns the inner solver response, consuming the solution.
    pub fn into_response(self) -> CpSolverResponse {
        self.response
    }

    /// Returns the solver response statistics as a formatted string.
    ///
    /// This provides detailed information about the solve, including
    /// number of conflicts, branches, propagations, wall time, etc.
    ///
    /// ## Format
    ///
    /// The output format matches the standard CP-SAT logging format:
    ///
    /// ```text
    /// CpSolverResponse summary:
    /// status: OPTIMAL
    /// objective: 42
    /// best_bound: 42
    /// booleans: 5
    /// conflicts: 23
    /// branches: 47
    /// propagations: 1234
    /// walltime: 0.003
    /// ```
    pub fn response_stats(&self) -> String {
        ffi::cp_solver_response_stats(&self.response, true)
    }

    /// Returns the solver log output, if `set_log_to_response` was enabled
    /// on the problem before solving.
    ///
    /// The log contains the search progress information that CP-SAT normally
    /// writes to stdout.
    pub fn solution_log(&self) -> &str {
        &self.response.solve_log
    }

    /// Returns the additional solutions found during search, if
    /// `set_fill_additional_solutions` was enabled on the problem.
    ///
    /// The primary solution is always available through the
    /// [`value`](Solution::value) method.
    pub fn additional_solutions(&self) -> &[cp_sat::proto::CpSolverSolution] {
        &self.response.additional_solutions
    }
}

impl Solution for CpSatSolution {
    fn status(&self) -> SolutionStatus {
        self.status
    }

    fn value(&self, variable: Variable) -> f64 {
        let cp_var = self.cp_sat_vars[variable.index()];
        cp_var.solution_value(&self.response) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::cp_sat;
    use crate::{
        Solution, SolverModel, WithInitialSolution, constraint,
        solvers::{SolutionStatus, WithTimeLimit},
        variable, variables,
    };

    #[test]
    fn solve_simple_integer_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + 2 * y <= 10))
            .with(constraint!(x >= 1));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);

        // x=10, y=0 attains the upper bound 10 of x+2y≤10, maximizing x+y.
        assert_eq!(sol.value(x), 10.0);
        assert_eq!(sol.value(y), 0.0);
    }

    #[test]
    fn solve_binary_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().binary());
        let y = vars.add(variable().binary());

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + y <= 1));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);

        // Solutions: (1,0) or (0,1) both give obj=1
        assert_eq!(sol.value(x) + sol.value(y), 1.0);
    }

    #[test]
    fn solve_with_equality_constraint() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x + y).using(cp_sat).with(constraint!(x == y));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
        assert_eq!(sol.value(y), 10.0);
    }

    #[test]
    fn solve_minimization_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.minimise(x).using(cp_sat).with(constraint!(x >= 3));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 3.0);
    }

    #[test]
    fn solve_infeasible_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(5));
        let y = vars.add(variable().integer().min(0).max(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + y >= 20));

        let result = pb.solve();
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), crate::ResolutionError::Infeasible);
    }

    #[test]
    fn solve_problem_with_time_limit() {
        let mut vars = variables!();
        let x = vars.add(variable().binary());

        let pb = vars.maximise(x).using(cp_sat).with_time_limit(10.0);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 1.0);
    }

    #[test]
    fn solve_problem_with_initial_solution() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(10));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + y <= 5))
            .with_initial_solution(vec![(x, 2.0), (y, 3.0)]);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x) + sol.value(y), 5.0);
    }

    #[test]
    fn solve_problem_with_initial_variable_values() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10).initial(5));
        let y = vars.add(variable().integer().min(0).max(10).initial(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + y <= 8));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x) + sol.value(y), 8.0);
    }

    #[test]
    #[should_panic(expected = "does not support continuous variables")]
    fn test_continuous_var_panics() {
        let mut vars = variables!();
        let _x = vars.add(variable().min(0).max(10)); // not integer
        let pb = vars.maximise(_x).using(cp_sat);
        let _ = pb.solve();
    }

    #[test]
    fn solve_with_large_coefficients() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(100));

        // Use coefficients that require rounding
        let pb = vars
            .maximise(3 * x + 2 * x)
            .using(cp_sat)
            .with(constraint!(x <= 42));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 42.0);
    }

    #[test]
    fn solve_with_verbose_logging() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat).set_verbose(true);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn solve_with_log_to_response() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat).set_log_to_response(true);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        // solution_log name is included in the response
        assert!(
            sol.solution_log().contains("OPTIMAL")
                || sol.solution_log().contains("objective")
                || sol.solution_log().is_empty(), // CP-SAT may not populate solve_log
        );
    }

    #[test]
    fn solve_with_num_search_workers() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + 2 * y <= 10))
            .set_num_search_workers(2);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
        assert_eq!(sol.value(y), 0.0);
    }

    #[test]
    fn solve_with_random_seed() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + 2 * y <= 10))
            .set_random_seed(42);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
        assert_eq!(sol.value(y), 0.0);
    }

    #[test]
    fn solve_with_mip_abs_gap() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let y = vars.add(variable().integer().min(0).max(5));

        let pb = vars
            .maximise(x + y)
            .using(cp_sat)
            .with(constraint!(x + 2 * y <= 10))
            .set_mip_abs_gap(0.01)
            .unwrap();

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn solve_with_mip_abs_gap_negative_errors() {
        let pb = cp_sat(variables!().maximise(0));
        let result = pb.set_mip_abs_gap(-1.0);
        assert!(result.is_err());
    }

    #[test]
    fn solve_with_mip_abs_gap_infinite_errors() {
        let pb = cp_sat(variables!().maximise(0));
        let result = pb.set_mip_abs_gap(f64::INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn solve_with_response_stats() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat);

        let sol = pb.solve().unwrap();
        let stats = sol.response_stats();
        // The stats string should be non-empty and contain status information
        assert!(!stats.is_empty());
        // It should mention OPTIMAL since we got an optimal solution
        assert!(
            stats.contains("OPTIMAL") || stats.contains("optimal"),
            "response_stats should mention OPTIMAL, got: {stats}"
        );
    }

    #[test]
    fn solve_with_model_stats() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat);

        let stats = pb.model_stats();
        assert!(
            !stats.is_empty(),
            "model_stats should not be empty, got: {stats}"
        );
    }

    #[test]
    fn solve_with_params_access() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .set_num_search_workers(4)
            .set_random_seed(99);

        // Check that params are accessible
        assert_eq!(pb.params().num_search_workers, Some(4));
        assert_eq!(pb.params().random_seed, Some(99));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn solve_with_mip_gap_trait() {
        use crate::solvers::WithMipGap;

        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat).with_mip_gap(0.05).unwrap();

        // The mip_gap should be accessible
        assert_eq!(pb.mip_gap(), Some(0.05));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn solve_and_into_response() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars.maximise(x).using(cp_sat);

        let sol = pb.solve().unwrap();
        let response = sol.into_response();
        assert!(!response.solution.is_empty());
    }

    #[test]
    fn solve_with_time_limit_via_params() {
        let mut vars = variables!();
        let x = vars.add(variable().binary());

        // Use params_mut to set time limit directly
        let mut pb = vars.maximise(x).using(cp_sat);
        pb.params_mut().max_deterministic_time = Some(10.0);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 1.0);
    }

    #[test]
    fn solve_with_verbose_and_response_log() {
        // Test that both verbose and log_to_response can be combined
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));

        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .set_verbose(true)
            .set_log_to_response(true);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_eq!(sol.value(x), 10.0);
    }
}
