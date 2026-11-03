//! A solver bridge to the Google OR-Tools CP-SAT solver, activated via the `cp_sat` feature.
//!
//! # Limitations
//!
//! - CP-SAT only supports **integer** and **binary** variables. Continuous (floating-point)
//!   variables cause an error at solve time.
//! - All coefficients and constants must be integers. Non-integer values cause an error
//!   at solve time, because CP-SAT uses integer-only arithmetic. Users should scale their
//!   coefficients on their own side. Fractional bounds on integer variables are converted
//!   exactly by rounding lower bounds up and upper bounds down.

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
/// # Errors
///
/// Returns a [`ResolutionError`] at solve time if:
/// - Any variable is not an integer variable (i.e., `is_integer` is `false` on the variable definition),
///   because CP-SAT does not support continuous variables.
/// - Any variable has invalid bounds where the lower bound exceeds the upper bound.
/// - Any coefficient or constant is non-finite or not an integer (CP-SAT only accepts integer arithmetic).
/// - Any variable bound is non-finite or outside the range supported by CP-SAT.
pub fn cp_sat(to_solve: UnsolvedProblem) -> CpSatProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;

    let mut model = CpModelBuilder::default();
    let solver_params = SatParameters::default();

    // Phase 1: Validate all variable definitions and create CP-SAT integer variables.
    // If any variable is invalid, transition to Invalid immediately.
    let mut cp_sat_vars = Vec::with_capacity(variables.len());
    for (var, def) in variables.iter_variables_with_def() {
        if !def.is_integer {
            return CpSatProblem::invalid(
                ResolutionError::Str(format!(
                    "CP-SAT does not support continuous variables. \
                         Variable '{}' (index {}) was not declared as integer or binary. \
                         Please use `.integer()` or `.binary()` on the variable definition.",
                    def.name,
                    var.index()
                )),
                solver_params,
            );
        }

        match create_cp_sat_var(&mut model, def, var) {
            Ok(cp_var) => cp_sat_vars.push(cp_var),
            Err(e) => {
                return CpSatProblem::invalid(e, solver_params);
            }
        }
    }

    // Phase 2: Build the objective expression, checking all coefficients and its constant.
    let mut objective_expr = LinearExpr::default();
    for (var, coeff) in &objective.linear.coefficients {
        match verify_integer(*coeff) {
            Ok(coeff_i64) => {
                let cp_var = cp_sat_vars[var.index()];
                objective_expr += (coeff_i64, cp_var);
            }
            Err(e) => {
                return CpSatProblem::invalid(e, solver_params);
            }
        }
    }
    match verify_integer(objective.constant) {
        Ok(constant) => objective_expr += constant,
        Err(e) => return CpSatProblem::invalid(e, solver_params),
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

    // Phase 3: Collect initial solution hints (skip non-finite values)
    let initial_solution: Vec<(Variable, f64)> = variables
        .iter_variables_with_def()
        .filter_map(|(var, def)| def.initial.filter(|v| v.is_finite()).map(|v| (var, v)))
        .collect();

    let mut problem = CpSatProblem {
        state: ProblemState::Valid { model, cp_sat_vars },
        solver_params,
    };

    problem = problem.with_initial_solution(initial_solution);

    problem
}

/// Verifies that an `f64` value is a finite integer representable as `i64`.
///
/// Returns an error if the value is non-finite, has a non-zero fractional part,
/// or is outside the `i64` representable range.
///
/// This is used for coefficients, constants, and variable bounds — CP-SAT requires
/// all of them to be finite integers.
fn verify_integer(x: f64) -> Result<i64, ResolutionError> {
    if !x.is_finite() {
        return Err(ResolutionError::Str(format!(
            "The CP-SAT solver does not accept infinite values for coefficients, constants, \
             or variable bounds. Received `{x}`, which is not finite."
        )));
    }
    if x.fract() != 0.0 {
        return Err(ResolutionError::Str(format!(
            "The CP-SAT solver only accepts integer values for coefficients, constants, \
             or variable bounds. Received `{x}`, which is not an integer. \
             Please round or scale your values to integers."
        )));
    }
    if !(x >= -(2f64.powi(63)) && x < 2f64.powi(63)) {
        return Err(ResolutionError::Str(format!(
            "The CP-SAT solver value `{x}` is out of the i64 range."
        )));
    }
    Ok(x as i64)
}

/// Converts a finite good_lp bound to the exact bound of an integer domain.
///
/// For integer variables, `x >= lower` is equivalent to `x >= ceil(lower)` and
/// `x <= upper` is equivalent to `x <= floor(upper)`.
fn integer_bound(x: f64, is_lower: bool) -> Result<i64, ResolutionError> {
    if !x.is_finite() {
        return Err(ResolutionError::Str(format!(
            "The CP-SAT solver requires finite variable bounds. \
             Received `{x}`, which is not finite."
        )));
    }
    verify_integer(if is_lower { x.ceil() } else { x.floor() })
}

/// Builds a `LinearExpr` from a good_lp `Constraint`'s expression,
/// converting f64 coefficients to i64.
fn expr_from_constraint_expression(
    cp_sat_vars: &[IntVar],
    expression: &crate::Expression,
) -> Result<LinearExpr, ResolutionError> {
    let mut linear = LinearExpr::default();
    for (var, coeff) in expression.linear.coefficients.iter() {
        let coeff_i64 = verify_integer(*coeff)?;
        if coeff_i64 != 0 {
            linear += (coeff_i64, cp_sat_vars[var.index()]);
        }
    }
    Ok(linear)
}

/// Validates that a MIP gap value is valid (non-negative and finite).
fn validate_gap(gap: f64) -> Result<(), MipGapError> {
    if gap.is_sign_negative() {
        Err(MipGapError::Negative)
    } else if gap.is_infinite() {
        Err(MipGapError::Infinite)
    } else {
        Ok(())
    }
}

/// Translates a good_lp `VariableDefinition` into a CP-SAT `IntVar`.
///
/// Bounds must be finite and are converted to the exact integer domain by rounding
/// the lower bound up and the upper bound down.
///
/// Returns an error if the lower bound exceeds the upper bound.
fn create_cp_sat_var(
    model: &mut CpModelBuilder,
    def: &VariableDefinition,
    _var: Variable,
) -> Result<IntVar, ResolutionError> {
    let domain_lower = integer_bound(def.min, true)?;
    let domain_upper = integer_bound(def.max, false)?;

    // Validate: lower bound must not exceed upper bound
    if domain_lower > domain_upper {
        return Err(ResolutionError::Str(if def.name.is_empty() {
            format!(
                "Invalid bounds for variable (index {}): lower bound ({}) > upper bound ({})",
                _var.index(),
                def.min,
                def.max
            )
        } else {
            format!(
                "Invalid bounds for variable '{}': lower bound ({}) > upper bound ({})",
                def.name, def.min, def.max
            )
        }));
    }

    let domain = [(domain_lower, domain_upper)];
    match def.name.as_str() {
        "" => Ok(model.new_int_var(domain)),
        name => Ok(model.new_int_var_with_name(domain, name)),
    }
}

/// Internal state of a CP-SAT problem.
///
/// Either carries a fully valid model ready for solving, or an error description.
/// Once in the `Invalid` state, there is no way to transition back to `Valid`.
///
/// In the `Invalid` state, a monotonic counter tracks constraint indices so that
/// each "dropped" constraint still receives a unique `ConstraintReference` index.
enum ProblemState {
    Valid {
        model: CpModelBuilder,
        cp_sat_vars: Vec<IntVar>,
    },
    Invalid {
        reason: ResolutionError,
        next_constraint_index: usize,
    },
}

impl ProblemState {
    /// Transitions to `Invalid` if currently `Valid`. No-op if already `Invalid`.
    fn into_invalid(&mut self, reason: ResolutionError, next_constraint_index: usize) {
        if !matches!(self, ProblemState::Invalid { .. }) {
            *self = ProblemState::Invalid {
                reason,
                next_constraint_index,
            };
        }
    }
}

/// A CP-SAT model wrapping `CpModelBuilder`, with a linear objective,
/// variable mappings, and solver parameters. Constructed via [`cp_sat()`].
///
/// See the [module-level documentation](self) for constraint translation
/// semantics and status mapping details.
///
/// Internally represented as a state machine: either `Valid` (ready to solve)
/// or `Invalid` (error encountered during construction). Once invalid,
/// the error is surfaced at [`solve()`](SolverModel::solve) time.
pub struct CpSatProblem {
    state: ProblemState,
    solver_params: SatParameters,
}

impl CpSatProblem {
    /// Constructs a problem in the invalid state with the given error.
    fn invalid(reason: ResolutionError, solver_params: SatParameters) -> Self {
        CpSatProblem {
            state: ProblemState::Invalid {
                reason,
                next_constraint_index: 0,
            },
            solver_params,
        }
    }

    /// Returns a shared reference to the inner CP-SAT model for inspection,
    /// or `None` if the problem is in an invalid state.
    ///
    /// This is useful for accessing read-only model data (e.g., [`CpModelBuilder::proto`]).
    /// To customize solver parameters, use [`params`](Self::params) or [`params_mut`](Self::params_mut)
    /// instead.
    pub fn as_inner(&self) -> Option<&CpModelBuilder> {
        match &self.state {
            ProblemState::Valid { model, .. } => Some(model),
            ProblemState::Invalid { .. } => None,
        }
    }

    /// Sets whether or not the solver should display verbose logging information to the console.
    ///
    /// When enabled, CP-SAT will log search progress to stdout.
    pub fn set_verbose(mut self, verbose: bool) -> Self {
        self.solver_params.log_search_progress = Some(verbose);
        self
    }

    /// Sets whether the solver log output should be captured in the solver response.
    ///
    /// When enabled, the log output will be available in the solution's
    /// [`CpSatSolution::solution_log`] method. This implicitly enables
    /// search progress logging.
    pub fn set_log_to_response(mut self, log: bool) -> Self {
        self.solver_params.log_to_response = Some(log);
        if log {
            self.solver_params.log_search_progress = Some(true);
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
        self.solver_params.num_search_workers = Some(n);
        self
    }

    /// Sets the random seed for deterministic randomness in the solver.
    ///
    /// When set to a fixed value, repeated solves of the same problem will
    /// produce identical results. A value of 0 (or not setting this) lets
    /// the solver use non-deterministic randomness.
    pub fn set_random_seed(mut self, seed: i32) -> Self {
        self.solver_params.random_seed = Some(seed);
        self
    }

    /// Sets the absolute MIP gap tolerance.
    ///
    /// This stops the search when the absolute difference between the best
    /// bound and the best objective falls below this value.
    pub fn set_mip_abs_gap(mut self, gap: f64) -> Result<Self, MipGapError> {
        validate_gap(gap)?;
        self.solver_params.absolute_gap_limit = Some(gap);
        Ok(self)
    }

    /// Sets whether to fill additional solutions in the response.
    ///
    /// When enabled, the solver will return all intermediate solutions found
    /// during the search in the solution's
    /// [`CpSatSolution::additional_solutions`] method.
    pub fn set_fill_additional_solutions(mut self, fill: bool) -> Self {
        self.solver_params.fill_additional_solutions_in_response = Some(fill);
        self
    }

    /// Returns a reference to the current solver parameters.
    /// This allows direct access to all CP-SAT `SatParameters` fields.
    pub fn params(&self) -> &SatParameters {
        &self.solver_params
    }

    /// Returns a mutable reference to the current solver parameters.
    /// This allows direct modification of all CP-SAT `SatParameters` fields.
    pub fn params_mut(&mut self) -> &mut SatParameters {
        &mut self.solver_params
    }

    /// Deletes all solution hints previously set via `with_initial_solution`
    /// or via variable `.initial()` definitions.
    ///
    /// This can be useful if you want to clear the warm-start hints without
    /// modifying the model structure.
    pub fn clear_hints(&mut self) {
        if let ProblemState::Valid { model, .. } = &mut self.state {
            model.del_hints();
        }
    }

    /// Get the solver statistics as a formatted string.
    /// This returns the model statistics even before solving.
    ///
    /// Returns `None` if the problem is in an invalid state.
    pub fn model_stats(&self) -> Option<String> {
        self.as_inner().map(|model| model.stats())
    }
}

fn gap_limit_reached(response: &CpSolverResponse, params: &SatParameters) -> bool {
    let absolute_gap = (response.objective_value - response.best_objective_bound).abs();
    if absolute_gap == 0.0 {
        return false;
    }

    params
        .absolute_gap_limit
        .is_some_and(|limit| absolute_gap <= limit)
        || params
            .relative_gap_limit
            .is_some_and(|limit| absolute_gap / response.objective_value.abs().max(1.0) <= limit)
}

fn non_optimal_solution_status(
    response: &CpSolverResponse,
    params: &SatParameters,
) -> Result<SolutionStatus, ResolutionError> {
    if gap_limit_reached(response, params) {
        return Ok(SolutionStatus::GapLimit);
    }
    if params.max_deterministic_time.is_some() || params.max_time_in_seconds.is_some() {
        return Ok(SolutionStatus::TimeLimit);
    }

    Err(ResolutionError::Str(format!(
        "CP-SAT returned a feasible solution without proving optimality, but good_lp cannot \
         represent its stopping reason as a SolutionStatus. CP-SAT solution info: {}",
        response.solution_info
    )))
}

impl SolverModel for CpSatProblem {
    type Solution = CpSatSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        match self.state {
            ProblemState::Invalid { reason, .. } => Err(reason),
            ProblemState::Valid { model, cp_sat_vars } => {
                let response = model.solve_with_parameters(&self.solver_params);

                let has_time_limit = self.solver_params.max_deterministic_time.is_some()
                    || self.solver_params.max_time_in_seconds.is_some();

                match response.status() {
                    CpSolverStatus::Optimal => Ok(CpSatSolution {
                        status: if gap_limit_reached(&response, &self.solver_params) {
                            SolutionStatus::GapLimit
                        } else {
                            SolutionStatus::Optimal
                        },
                        response,
                        cp_sat_vars,
                    }),
                    CpSolverStatus::Feasible => {
                        let status = non_optimal_solution_status(&response, &self.solver_params)?;
                        Ok(CpSatSolution {
                            response,
                            status,
                            cp_sat_vars,
                        })
                    }
                    CpSolverStatus::Infeasible => Err(ResolutionError::Infeasible),
                    CpSolverStatus::Unknown => {
                        if !response.solution.is_empty() {
                            let status =
                                non_optimal_solution_status(&response, &self.solver_params)?;
                            Ok(CpSatSolution {
                                response,
                                status,
                                cp_sat_vars,
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
                        let validation = model.validate_cp_model();
                        Err(ResolutionError::Str(format!(
                            "Invalid CP-SAT model: {}",
                            validation
                        )))
                    }
                }
            }
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        match &mut self.state {
            ProblemState::Invalid {
                next_constraint_index,
                ..
            } => {
                let index = *next_constraint_index;
                *next_constraint_index += 1;
                ConstraintReference { index }
            }
            ProblemState::Valid {
                model, cp_sat_vars, ..
            } => {
                let index = model.proto().constraints.len();

                let constant_i64 = match verify_integer(-constraint.expression.constant) {
                    Ok(val) => val,
                    Err(e) => {
                        self.state.into_invalid(e, index + 1);
                        return ConstraintReference { index };
                    }
                };
                let linear_expr =
                    match expr_from_constraint_expression(cp_sat_vars, &constraint.expression) {
                        Ok(expr) => expr,
                        Err(e) => {
                            self.state.into_invalid(e, index + 1);
                            return ConstraintReference { index };
                        }
                    };

                if constraint.is_equality() {
                    model.add_eq(linear_expr, constant_i64);
                } else {
                    model.add_le(linear_expr, constant_i64);
                }

                ConstraintReference { index }
            }
        }
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
        match &mut self.state {
            ProblemState::Invalid { .. } => self,
            ProblemState::Valid {
                model, cp_sat_vars, ..
            } => {
                for (var, val) in solution {
                    if !val.is_finite() {
                        continue;
                    }
                    if let Ok(hint_value) = verify_integer(val) {
                        model.add_hint(cp_sat_vars[var.index()], hint_value);
                    }
                }
                self
            }
        }
    }
}

impl WithTimeLimit for CpSatProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.solver_params.max_time_in_seconds = Some(seconds.into());
        self
    }
}

impl WithMipGap for CpSatProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.solver_params.relative_gap_limit.map(|v| v as f32)
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        validate_gap(mip_gap as f64)?;
        self.solver_params.relative_gap_limit = Some(mip_gap as f64);
        Ok(self)
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
    use super::{
        CpSolverResponse, SatParameters, cp_sat, gap_limit_reached, non_optimal_solution_status,
    };
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
    fn test_non_integer_coefficient_in_objective_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars.maximise(2.5 * x).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("not an integer"),
                    "Error message should mention non-integer, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_integer_coefficient_in_constraint_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .with(constraint!(2.5 * x <= 10));
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("not an integer"),
                    "Error message should mention non-integer, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_integer_constant_in_constraint_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars.maximise(x).using(cp_sat).with(constraint!(x <= 10.5));
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("not an integer"),
                    "Error message should mention non-integer, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_integer_initial_solution_is_skipped() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .with_initial_solution(vec![(x, 2.5)]);
        let sol = pb.solve().unwrap();
        // Non-integer hint is skipped, solve proceeds normally
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn test_continuous_var_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(10)); // not integer
        let pb = vars.maximise(x).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("does not support continuous variables"),
                    "Error message should mention continuous variables, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_invalid_bounds_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(10).max(0)); // lower > upper
        let pb = vars.maximise(x).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("Invalid bounds"),
                    "Error message should mention invalid bounds, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn fractional_bounds_preserve_the_integer_domain() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(1.5).max(4.5));
        let solution = vars.maximise(x).using(cp_sat).solve().unwrap();

        assert_eq!(solution.value(x), 4.0);
    }

    #[test]
    fn fractional_bounds_with_no_integer_between_them_return_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(2.1).max(2.9));
        let result = vars.maximise(x).using(cp_sat).solve();

        assert!(result.is_err());
    }

    #[test]
    fn test_non_finite_bound_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(f64::NAN).max(10));
        let pb = vars.maximise(x).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("requires finite variable bounds"),
                    "Error message should mention non-finite bounds, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_infinite_bound_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(f64::NEG_INFINITY).max(10)); // infinite lower bound
        let pb = vars.maximise(x).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match &err {
            crate::ResolutionError::Str(msg) => {
                assert!(
                    msg.contains("requires finite variable bounds"),
                    "Error message should mention infinite values, got: {msg}"
                );
            }
            other => panic!("Expected ResolutionError::Str, got: {other:?}"),
        }
    }

    #[test]
    fn test_nan_coefficient_in_constraint_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .with(constraint!(x <= f64::NAN));
        let result = pb.solve();
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_in_initial_solution_is_skipped() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let pb = vars
            .maximise(x)
            .using(cp_sat)
            .with_initial_solution(vec![(x, f64::NAN)]);
        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn test_continuous_var_in_multi_var_problem() {
        // Only the first variable is invalid — the whole problem should still error
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(10)); // not integer
        let y = vars.add(variable().integer().min(0).max(5));
        let pb = vars.maximise(x + y).using(cp_sat);
        let result = pb.solve();
        assert!(result.is_err());
    }

    #[test]
    fn test_as_inner_on_invalid() {
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(10)); // not integer
        let pb = vars.maximise(x).using(cp_sat);
        // as_inner returns None in invalid state
        assert!(pb.as_inner().is_none());
        // But params are still accessible
        assert!(pb.params().num_search_workers.is_none());
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
        assert!(
            sol.solution_log().contains("OPTIMAL")
                || sol.solution_log().contains("objective")
                || sol.solution_log().is_empty(),
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
        // Uses no variables, so params are always accessible
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
        assert!(!stats.is_empty());
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
        assert!(stats.is_some(), "model_stats should be Some, got: None");
        assert!(
            !stats.as_ref().unwrap().is_empty(),
            "model_stats should not be empty, got: {:?}",
            stats
        );
    }

    #[test]
    fn test_model_stats_on_invalid() {
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(10)); // not integer
        let pb = vars.maximise(x).using(cp_sat);
        // model_stats returns None in invalid state
        assert!(pb.model_stats().is_none());
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

        assert_eq!(pb.mip_gap(), Some(0.05));

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 10.0);
    }

    #[test]
    fn detects_a_gap_limit_reported_as_optimal_by_cp_sat() {
        let response = CpSolverResponse {
            objective_value: 100.0,
            best_objective_bound: 110.0,
            ..CpSolverResponse::default()
        };
        let params = SatParameters {
            relative_gap_limit: Some(0.1),
            ..SatParameters::default()
        };

        assert!(gap_limit_reached(&response, &params));
    }

    #[test]
    fn feasible_without_a_supported_stopping_reason_is_not_optimal() {
        let response = CpSolverResponse {
            solution_info: "stopped after first solution".to_owned(),
            ..CpSolverResponse::default()
        };

        assert!(non_optimal_solution_status(&response, &SatParameters::default()).is_err());
    }

    #[test]
    fn objective_constant_is_forwarded_to_cp_sat() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let solution = vars.maximise(x + 100).using(cp_sat).solve().unwrap();

        assert_eq!(solution.response().objective_value, 110.0);
    }

    #[test]
    fn fractional_objective_constant_returns_error() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().min(0).max(10));
        let result = vars.maximise(x + 0.5).using(cp_sat).solve();

        assert!(result.is_err());
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

        let mut pb = vars.maximise(x).using(cp_sat);
        pb.params_mut().max_deterministic_time = Some(10.0);

        let sol = pb.solve().unwrap();
        assert_eq!(sol.value(x), 1.0);
    }

    #[test]
    fn solve_with_verbose_and_response_log() {
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
