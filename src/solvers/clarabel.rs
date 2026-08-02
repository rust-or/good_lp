//! A solver that uses [clarabel](https://oxfordcontrol.github.io/ClarabelDocs/stable/), a pure rust solver.

use crate::affine_expression_trait::IntoAffineExpression;
use crate::expression::LinearExpression;
use crate::variable::UnsolvedProblem;
use crate::{Constraint, DualValues, SolutionWithDual, Variable};
use crate::{
    SolutionStatus,
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
};

use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT::{self, *};
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::{DefaultSolution, SolverStatus};
use clarabel::solver::{DefaultSolver, IPSolver};

/// The [clarabel](https://oxfordcontrol.github.io/ClarabelDocs/stable/) solver,
/// to be used with [UnsolvedProblem::using].
pub fn clarabel(to_solve: UnsolvedProblem) -> ClarabelProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let objective_factor = if direction == ObjectiveDirection::Maximisation {
        -1.
    } else {
        1.
    };
    let mut objective_vector = vec![0.; variables.len()];
    for (var, obj) in objective.linear_coefficients() {
        objective_vector[var.index()] = obj * objective_factor;
    }
    let constraints_matrix_builder = CscMatrixBuilder::new(variables.len());
    let mut settings = DefaultSettingsBuilder::default();
    settings.verbose(false).tol_feas(1e-9);
    let mut p = ClarabelProblem {
        objective: objective_vector,
        constraints_matrix_builder,
        constraint_values: Vec::new(),
        shadow_price_scales: Vec::new(),
        // Clarabel's standard form is
        //
        //     minimize g(x), subject to A*x + s = b
        //
        // (https://clarabel.org/stable/rust/getting_started_rs/#problem-format),
        // and `DefaultSolution::z` is the dual solution
        // (https://docs.rs/clarabel/0.11.0/clarabel/solver/implementations/default/struct.DefaultSolution.html#structfield.z).
        // Thus the Lagrangian contains `z' * (A*x + s - b)`, so the
        // sensitivity of Clarabel's objective to `b` is `d g*/d b = -z`.
        // Since `g = objective_factor * f`, the objective part of the
        // conversion from `z` to good_lp's shadow price is
        // `-objective_factor`. The row's RHS transformation is composed with
        // this factor in `add_constraint` below.
        objective_shadow_price_scale: -objective_factor,
        variables: variables.len(),
        settings,
        cones: Vec::new(),
    };
    // add trivial constraints embedded in the variable definitions
    for (var, def) in variables.iter_variables_with_def() {
        if def.is_integer {
            panic!("Clarabel doesn't support integer variables")
        }
        if def.min != f64::NEG_INFINITY {
            p.add_constraint(var >> def.min);
        }
        if def.max != f64::INFINITY {
            p.add_constraint(var << def.max);
        }
    }
    p
}

/// A clarabel model
pub struct ClarabelProblem {
    constraints_matrix_builder: CscMatrixBuilder,
    constraint_values: Vec<f64>,
    /// Final multiplier from each Clarabel dual `z_i` to good_lp's shadow price.
    shadow_price_scales: Vec<f64>,
    /// Objective-only part of that multiplier; the row part is applied when
    /// the constraint is added.
    objective_shadow_price_scale: f64,
    objective: Vec<f64>,
    variables: usize,
    settings: DefaultSettingsBuilder<f64>,
    cones: Vec<SupportedConeT<f64>>,
}

impl ClarabelProblem {
    /// Access the problem settings
    pub fn settings(&mut self) -> &mut DefaultSettingsBuilder<f64> {
        &mut self.settings
    }

    /// Convert the problem into a clarabel solver.
    /// Panics if the problem is not valid.
    pub fn into_solver(self) -> DefaultSolver<f64> {
        self.try_into_solver()
            .expect("Invalid clarabel problem. This is likely a bug in good_lp. Problems should always have coherent dimensions.")
    }

    /// Convert the problem into a clarabel solver.
    pub fn try_into_solver(self) -> Result<DefaultSolver<f64>, ResolutionError> {
        let settings = self
            .settings
            .build()
            .map_err(|e| ResolutionError::Str(format!("Invalid clarabel settings: {e}")))?;

        let quadratic_objective = &CscMatrix::zeros((self.variables, self.variables));
        let objective = &self.objective;
        let constraints = &self.constraints_matrix_builder.build();
        let constraint_values = &self.constraint_values;
        let cones = &self.cones;

        DefaultSolver::new(
            quadratic_objective,
            objective,
            constraints,
            constraint_values,
            cones,
            settings,
        )
        .map_err(|error| ResolutionError::Str(error.to_string()))
    }
}

impl SolverModel for ClarabelProblem {
    type Solution = ClarabelSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let mut problem = self;
        let shadow_price_scales = std::mem::take(&mut problem.shadow_price_scales);
        let mut solver = problem.try_into_solver()?;
        solver.solve();
        match solver.solution.status {
            SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
                Err(ResolutionError::Infeasible)
            }
            SolverStatus::Solved
            | SolverStatus::AlmostSolved
            | SolverStatus::AlmostDualInfeasible
            | SolverStatus::DualInfeasible => Ok(ClarabelSolution {
                solution: solver.solution,
                shadow_price_scales,
            }),
            SolverStatus::Unsolved => Err(ResolutionError::Other("Unsolved")),
            SolverStatus::MaxIterations => Err(ResolutionError::Other("Max iterations reached")),
            SolverStatus::MaxTime => Err(ResolutionError::Other("Time limit reached")),
            SolverStatus::NumericalError => Err(ResolutionError::Other("Numerical error")),
            SolverStatus::InsufficientProgress => Err(ResolutionError::Other("No progress")),
            SolverStatus::CallbackTerminated => Err(ResolutionError::Other("Callback terminated")),
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let is_equality = constraint.is_equality();
        let is_greater_than_or_equal = constraint.is_greater_than_or_equal();
        self.constraints_matrix_builder
            .add_row(constraint.expression.linear);
        let index = self.constraint_values.len();
        self.constraint_values.push(-constraint.expression.constant);
        // good_lp stores `lhs >= rhs` as `rhs - lhs <= 0`. Consequently the
        // corresponding Clarabel bound `b` changes by -1 when the user-written
        // `rhs` changes by +1. For `<=` and `==` rows it changes by +1. Compose
        // that `d b/d rhs` with the objective scale derived in `clarabel` so
        // dual retrieval only has to multiply by the completed conversion.
        let rhs_scale = if is_greater_than_or_equal { -1. } else { 1. };
        self.shadow_price_scales
            .push(self.objective_shadow_price_scale * rhs_scale);
        // Cones indicate the type of constraint. We only support nonnegative and equality constraints.
        // To avoid creating a new cone for each constraint, we merge them.
        let next_cone = if is_equality {
            ZeroConeT(1)
        } else {
            NonnegativeConeT(1)
        };
        let prev_cone = self.cones.last_mut();
        match (prev_cone, next_cone) {
            (Some(ZeroConeT(a)), ZeroConeT(b)) => *a += b,
            (Some(NonnegativeConeT(a)), NonnegativeConeT(b)) => *a += b,
            (_, next_cone) => self.cones.push(next_cone),
        };
        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Clarabel"
    }
}

/// The solution to a clarabel problem
pub struct ClarabelSolution {
    solution: DefaultSolution<f64>,
    shadow_price_scales: Vec<f64>,
}

impl ClarabelSolution {
    /// Returns the clarabel solution object. You can use it to dynamically add new constraints
    pub fn into_inner(self) -> DefaultSolution<f64> {
        self.solution
    }

    /// Borrow the clarabel solution object
    pub fn inner(&self) -> &DefaultSolution<f64> {
        &self.solution
    }
}

impl Solution for ClarabelSolution {
    fn status(&self) -> SolutionStatus {
        SolutionStatus::Optimal
    }
    fn value(&self, variable: Variable) -> f64 {
        self.solution.x[variable.index()]
    }
}

impl<'a> SolutionWithDual<'a> for ClarabelSolution {
    type Dual = &'a ClarabelSolution;

    fn compute_dual(&'a mut self) -> Self::Dual {
        self
    }
}

impl DualValues for &ClarabelSolution {
    fn dual(&self, constraint: ConstraintReference) -> f64 {
        self.solution.z[constraint.index] * self.shadow_price_scales[constraint.index]
    }
}

struct CscMatrixBuilder {
    /// Indicates the row index of the corresponding element in `nzval`
    rowval: Vec<Vec<usize>>,
    /// All non-zero values in the matrix, in column-major order
    nzval: Vec<Vec<f64>>,
    n_rows: usize,
    n_cols: usize,
}

impl CscMatrixBuilder {
    fn new(n_cols: usize) -> Self {
        Self {
            rowval: vec![Vec::new(); n_cols],
            nzval: vec![Vec::new(); n_cols],
            n_rows: 0,
            n_cols,
        }
    }
    fn add_row(&mut self, row: LinearExpression) {
        for (var, value) in row.linear_coefficients() {
            self.rowval[var.index()].push(self.n_rows);
            self.nzval[var.index()].push(value);
        }
        self.n_rows += 1;
    }
    fn build(self) -> clarabel::algebra::CscMatrix {
        let mut colptr = Vec::with_capacity(self.n_cols + 1);
        colptr.push(0);
        for col in &self.rowval {
            colptr.push(colptr.last().unwrap() + col.len());
        }
        clarabel::algebra::CscMatrix::new(
            self.n_rows,
            self.n_cols,
            colptr,
            fast_flatten_vecs(self.rowval),
            fast_flatten_vecs(self.nzval),
        )
    }
}

fn fast_flatten_vecs<T: Copy>(vecs: Vec<Vec<T>>) -> Vec<T> {
    // This is faster than vecs.into_iter().flatten().collect()
    // because it doesn't need to allocate a new Vec
    // (we take ownership of the first Vec and add the rest to it)
    let size: usize = vecs.iter().map(|v| v.len()).sum();
    let mut iter = vecs.into_iter();
    let mut result = if let Some(v) = iter.next() {
        v
    } else {
        return Vec::new();
    };
    result.reserve_exact(size - result.len());
    for v in iter {
        result.extend_from_slice(&v);
    }
    result
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::variables;

    #[test]
    fn test_csc_matrix_builder() {
        variables! {vars:
            x;
            y;
            z;
        }
        let mut builder = CscMatrixBuilder::new(3);
        builder.add_row((y + 2 * z).linear);
        builder.add_row((3 * x + 4 * y + 5 * z).linear);
        let matrix = builder.build();
        /* The matrix is:
        [ 0 1 2 ]
        [ 3 4 5 ]
        */
        assert_eq!(matrix.m, 2); // 2 rows
        assert_eq!(matrix.n, 3); // 3 columns
        assert_eq!(matrix.get_entry((0, 0)), None); // get_entry((row, col))
        assert_eq!(matrix.get_entry((0, 1)), Some(1.));
        assert_eq!(matrix.get_entry((0, 2)), Some(2.));
        assert_eq!(matrix.get_entry((1, 0)), Some(3.));
        assert_eq!(matrix.get_entry((1, 1)), Some(4.));
        assert_eq!(matrix.get_entry((1, 2)), Some(5.));
    }
}
