//! A solver that uses the [`pumpkin_solver`](https://docs.rs/pumpkin_solver) crate,
//! a pure-Rust SAT+CP solver that supports integral constraints and branching.
//!
//! **Important**:
//! - Pumpkin only supports *integral* variables, constraints, and objective coefficients.
//! - We panic if a user tries non-integer bounds, non-integer coefficients, or infinite bounds.
//! - We do not currently handle multi-solution enumeration or advanced time-limits, etc.

use pumpkin_solver::constraints;
use pumpkin_solver::results::{
    OptimisationResult,
    ProblemSolution, // to call .get_integer_value(...)
    SatisfactionResult,
    Solution as PumpkinRawSolution,
};
use pumpkin_solver::termination::Indefinite;
use pumpkin_solver::variables::{DomainId, TransformableVariable};
use pumpkin_solver::Solver as PumpkinSolver;

use crate::solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel};
use crate::IntoAffineExpression;
use crate::{
    constraint::ConstraintReference,
    expression::Expression,
    variable::{UnsolvedProblem, Variable},
    Constraint,
};
use crate::{CardinalityConstraintSolver, ModelWithSOS1};

/// Construct a [`PumpkinProblem`] from a [`UnsolvedProblem`].
/// You typically call:
///
/// ```ignore
/// # use good_lp::*;
/// use good_lp::solvers::pumpkin_solver::pumpkin;
///
/// let solution = vars.maximise(a + b)
///     .using(pumpkin)
///     .with(constraint!(a + b <= 10))
///     .solve()?;
/// ```
pub fn pumpkin(to_solve: UnsolvedProblem) -> PumpkinProblem {
    PumpkinProblem::new(to_solve)
}

/// A [`SolverModel`](crate::solvers::SolverModel) implemented using Pumpkin.
///
/// Because Pumpkin only handles integer variables/coefficients, we:
/// - Panic on non-integer bounds or coefficients
/// - Panic on ±∞ bounds
///
/// For the objective, we create a hidden `obj_var` that must match
/// the user-provided linear expression, then do `maximise(obj_var)` or `minimise(obj_var)`.
pub struct PumpkinProblem {
    /// The underlying Pumpkin solver
    solver: PumpkinSolver,
    /// For each good_lp variable, which Pumpkin domain ID we use
    domain_ids: Vec<DomainId>,

    /// Whether we want to do max or min
    direction: ObjectiveDirection,
    /// The user’s entire objective expression
    objective: Expression,

    /// Next index for constraints
    next_constraint_index: usize,
}

impl PumpkinProblem {
    /// Create a new Pumpkin problem from `UnsolvedProblem`.
    fn new(to_solve: UnsolvedProblem) -> Self {
        let UnsolvedProblem {
            objective,
            direction,
            variables,
        } = to_solve;

        let mut solver = PumpkinSolver::default();
        let mut domain_ids = Vec::with_capacity(variables.len());

        // create an integer domain for each good_lp variable
        for (_, def) in variables.iter_variables_with_def() {
            // must be integer
            if !def.is_integer {
                panic!(
                    "Pumpkin solver only supports integer variables, but got a non-integer definition: {:?}",
                    def.name
                );
            }

            // must have finite integral bounds
            if def.min.is_infinite() || def.max.is_infinite() {
                panic!(
                    "Pumpkin solver does not handle infinite bounds for variable {:?}",
                    def.name
                );
            }
            let lb_i32 = i32_or_panic(def.min, "variable lower bound");
            let ub_i32 = i32_or_panic(def.max, "variable upper bound");

            let d_id = solver.new_bounded_integer(lb_i32, ub_i32);
            domain_ids.push(d_id);
        }

        PumpkinProblem {
            solver,
            domain_ids,
            direction,
            objective,
            next_constraint_index: 0,
        }
    }

    /// Return an immutable reference to the underlying Pumpkin solver
    pub fn as_inner(&self) -> &PumpkinSolver {
        &self.solver
    }

    /// Return a mutable reference to the underlying Pumpkin solver
    /// (use with caution, as changing the structure can break good_lp assumptions).
    pub fn as_inner_mut(&mut self) -> &mut PumpkinSolver {
        &mut self.solver
    }
}

impl SolverModel for PumpkinProblem {
    type Solution = PumpkinSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        let mut brancher = self
            .solver
            .default_brancher_over_all_propositional_variables();
        let mut termination = Indefinite;

        // If the user’s objective is entirely zero, or if the user only has constant=0,
        // we can just do a "satisfy" approach
        if self.objective.linear.coefficients.is_empty() && self.objective.constant.abs() < 1e-9 {
            // no real objective => just satisfy
            let sat_result = self.solver.satisfy(&mut brancher, &mut termination);
            match sat_result {
                SatisfactionResult::Satisfiable(soln) => {
                    return Ok(PumpkinSolution {
                        solution: soln,
                        domain_ids: self.domain_ids,
                    });
                }
                SatisfactionResult::Unsatisfiable => {
                    return Err(ResolutionError::Infeasible);
                }
                SatisfactionResult::Unknown => {
                    return Err(ResolutionError::Other(
                        "Pumpkin: Unknown for feasibility check",
                    ));
                }
            }
        }

        // Otherwise, create an "obj_var" domain to represent the linear objective expression,
        // then do solver.maximise(obj_var) or solver.minimise(obj_var).
        let (obj_var, offset) = self.build_objective_var();

        // Add a constraint:  obj_var = sum_i (coeff_i * var_i) + offset
        // i.e.  obj_var - Σ(coeff_i * domain_i) = offset
        // => "obj_var - sum(...)= offset"
        let mut lhs_terms = Vec::new();
        lhs_terms.push(obj_var.scaled(1)); // + obj_var

        // subtract each coefficient × domain
        for (var, coeff) in &self.objective.linear.coefficients {
            let i_coeff = i32_or_panic(*coeff, "objective coefficient");
            lhs_terms.push(self.domain_ids[var.index()].scaled(-i_coeff));
        }

        let eq_constraint = constraints::equals(lhs_terms, offset);
        let _ = self.solver.add_constraint(eq_constraint).post();

        // Now do the actual call to `maximise(obj_var)` or `minimise(obj_var)`.
        let result = match self.direction {
            ObjectiveDirection::Maximisation => {
                self.solver
                    .maximise(&mut brancher, &mut termination, obj_var)
            }
            ObjectiveDirection::Minimisation => {
                self.solver
                    .minimise(&mut brancher, &mut termination, obj_var)
            }
        };

        match result {
            OptimisationResult::Optimal(sol) | OptimisationResult::Satisfiable(sol) => {
                // Satisfiable might not be proven optimal, but it is a valid solution
                Ok(PumpkinSolution {
                    solution: sol,
                    domain_ids: self.domain_ids,
                })
            }
            OptimisationResult::Unsatisfiable => Err(ResolutionError::Infeasible),
            OptimisationResult::Unknown => Err(ResolutionError::Other(
                "Pumpkin returned Unknown (no proof of optimality or infeasibility)",
            )),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.next_constraint_index;
        self.next_constraint_index += 1;

        // sum_i (coeff_i * var_i) + constant <= 0 or = 0
        // => sum_i (coeff_i * var_i) <= -constant
        // => sum_i (coeff_i) * domain_i <= rhs_i
        let rhs = i32_or_panic(-constraint.expression.constant, "constraint constant");
        let mut lhs_terms = Vec::new();
        for (&var, &coeff) in &constraint.expression.linear.coefficients {
            let i_coeff = i32_or_panic(coeff, "constraint coefficient");
            lhs_terms.push(self.domain_ids[var.index()].scaled(i_coeff));
        }

        if constraint.is_equality {
            let _ = self
                .solver
                .add_constraint(constraints::equals(lhs_terms, rhs))
                .post();
        } else {
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(lhs_terms, rhs))
                .post();
        }

        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Pumpkin Solver"
    }
}

impl PumpkinProblem {
    /// Create an integer variable for the objective expression
    /// plus an integer offset for the objective constant.
    fn build_objective_var(&mut self) -> (DomainId, i32) {
        // offset is the integer portion of self.objective.constant
        let offset = i32_or_panic(self.objective.constant, "objective constant");
        let lb = -1000;
        let ub = 1000;
        let obj_var = self.solver.new_bounded_integer(lb, ub);
        (obj_var, offset)
    }
}

/// A solution from the Pumpkin solver
pub struct PumpkinSolution {
    /// The actual Pumpkin solution
    solution: PumpkinRawSolution,
    /// The DomainId for each variable
    domain_ids: Vec<DomainId>,
}

impl Solution for PumpkinSolution {
    fn value(&self, variable: Variable) -> f64 {
        let domain_id = self.domain_ids[variable.index()];
        self.solution.get_integer_value(domain_id) as f64
    }
}

// Optional: implement SOS1 or cardinality constraints

impl ModelWithSOS1 for PumpkinProblem {
    fn add_sos1<I: IntoAffineExpression>(&mut self, vars_weights: I) {
        // Pumpkin does not have a direct SOS1 concept. We must approximate.
        // For example, force all these variables to be in [0..1], then sum <= 1.
        // If you do not truly need an SOS1, you can leave this unimplemented or panic!:
        let pair_list = vars_weights
            .linear_coefficients()
            .into_iter()
            .collect::<Vec<_>>();
        if pair_list.is_empty() {
            return;
        }

        // For each var, ensure domain range is [0,1].
        // Then add sum <= 1.
        let mut domain_ids = Vec::new();
        for (var, _) in pair_list {
            let domain_id = self.domain_ids[var.index()];
            // Constrain domain_id <= 1, domain_id >= 0
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(vec![domain_id], 1))
                .post();
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(
                    vec![domain_id.scaled(-1)],
                    0,
                ))
                .post();
            domain_ids.push(domain_id);
        }

        let _ = self
            .solver
            .add_constraint(constraints::less_than_or_equals(domain_ids, 1))
            .post();
    }
}

impl CardinalityConstraintSolver for PumpkinProblem {
    fn add_cardinality_constraint(&mut self, vars: &[Variable], rhs: usize) -> ConstraintReference {
        let index = self.next_constraint_index;
        self.next_constraint_index += 1;

        // Force each in [0,1], then sum <= rhs
        let mut domain_ids = Vec::new();
        for &v in vars {
            let d_id = self.domain_ids[v.index()];
            // ensure 0 <= d_id <= 1
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(vec![d_id], 1))
                .post();
            let _ = self
                .solver
                .add_constraint(constraints::less_than_or_equals(vec![d_id.scaled(-1)], 0))
                .post();
            domain_ids.push(d_id);
        }

        // sum_i domain_i <= rhs
        let _ = self
            .solver
            .add_constraint(constraints::less_than_or_equals(domain_ids, rhs as i32))
            .post();

        ConstraintReference { index }
    }
}

// -------------------------------------------------------------------
//  f64 -> i32 conversion with integral checks
// -------------------------------------------------------------------
fn i32_or_panic(value: f64, context: &str) -> i32 {
    if !value.is_finite() {
        panic!("Pumpkin solver: {} is infinite or NaN: {}", context, value);
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        panic!(
            "Pumpkin solver: {} must be integral, got: {}",
            context, value
        );
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        panic!(
            "Pumpkin solver: {} is out of i32 range, got: {}",
            context, rounded
        );
    }
    rounded as i32
}

// -------------------------------------------------------------------
//  Some basic tests
// -------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::pumpkin;
    use crate::{constraint, variable, variables, Solution, SolverModel};

    #[test]
    fn can_solve_simple_problem() {
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(12).integer());
        let y = vars.add(variable().min(0).max(12).integer());

        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(constraint!(x + y == 12))
            .solve()
            .unwrap();

        // Because we do "maximise(x+y)" subject to x+y=12, the solver might pick x=12,y=0
        // or y=12,x=0, etc. We only check that x+y==12
        let sum = solution.value(x) + solution.value(y);
        assert!((sum - 12.0).abs() <= 1e-9, "x+y must be 12, got {}", sum);
    }

    #[test]
    fn can_solve_easy() {
        let mut vars = variables!();
        let x = vars.add(variable().min(0).max(12).integer());
        let y = vars.add(variable().min(0).max(12).integer());
        let solution = vars
            .maximise(x + y)
            .using(pumpkin)
            .with(x + y << 12) // x + y <=12
            .solve()
            .unwrap();
        assert!(solution.value(x) + solution.value(y) <= 12.0 + 1e-9);
    }

    #[test]
    fn can_solve_milp() {
        let mut vars = variables!();
        let x = vars.add(variable().integer().clamp(0, 10));
        let y = vars.add(variable().integer().clamp(0, 10));

        let model = vars
            .maximise(3 * x + 2 * y)
            .using(pumpkin)
            .with(2 * x + 3 * y << 12)
            .with(x + y << 8);

        let sol = model.solve().expect("Should solve a small MILP");
        // check constraints
        assert!(2.0 * sol.value(x) + 3.0 * sol.value(y) <= 12.0 + 1e-9);
        assert!(sol.value(x) + sol.value(y) <= 8.0 + 1e-9);

        // check objective is at least feasible
        let obj = 3.0 * sol.value(x) + 2.0 * sol.value(y);
        assert!(obj >= 10.0);
    }
}
