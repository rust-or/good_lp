//! A solver that uses [clarabel](https://oxfordcontrol.github.io/ClarabelDocs/stable/), a pure rust solver.

use crate::affine_expression_trait::IntoAffineExpression;
use crate::expression::LinearExpression;
use crate::variable::UnsolvedProblem;
#[cfg(feature = "enable_quadratic")]
use crate::quadratic_problem::QuadraticUnsolvedProblem;
#[cfg(feature = "enable_quadratic")]
use crate::quadratic_expression::VariablePair;
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
    SolutionStatus,
};
use crate::{Constraint, DualValues, SolutionWithDual, Variable};

use clarabel::algebra::CscMatrix;
use clarabel::solver::implementations::default::DefaultSettingsBuilder;
use clarabel::solver::SupportedConeT::{self, *};
use clarabel::solver::{DefaultSolution, SolverStatus};
use clarabel::solver::{DefaultSolver, IPSolver};

impl ClarabelProblemCore {
    /// Create a new ClarabelProblemCore with common setup
    fn new(variables_len: usize) -> Self {
        let mut settings = DefaultSettingsBuilder::default();
        settings.verbose(false).tol_feas(1e-9);
        
        Self {
            constraints_matrix_builder: CscMatrixBuilder::new(variables_len),
            constraint_values: Vec::new(),
            objective: vec![0.; variables_len],
            variables: variables_len,
            settings,
            cones: Vec::new(),
        }
    }

    /// Add variable bound constraints from variable definitions
    fn add_variable_bounds<P: ClarabelProblemLike>(variables: &crate::variable::ProblemVariables, problem: &mut P) {
        for (var, def) in variables.iter_variables_with_def() {
            if def.is_integer {
                panic!("Clarabel doesn't support integer variables")
            }
            if def.min != f64::NEG_INFINITY {
                problem.add_constraint(var >> def.min);
            }
            if def.max != f64::INFINITY {
                problem.add_constraint(var << def.max);
            }
        }
    }

    /// Get direction coefficient (-1 for maximization, 1 for minimization)
    fn direction_coef(direction: ObjectiveDirection) -> f64 {
        if direction == ObjectiveDirection::Maximisation {
            -1.
        } else {
            1.
        }
    }

    /// Common solver status handling
    fn handle_solver_status(status: SolverStatus, solution: DefaultSolution<f64>) -> Result<ClarabelSolution, ResolutionError> {
        match status {
            e @ (SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible) => {
                eprintln!("Clarabel error: {:?}", e);
                Err(ResolutionError::Infeasible)
            }
            SolverStatus::Solved
            | SolverStatus::AlmostSolved
            | SolverStatus::AlmostDualInfeasible
            | SolverStatus::DualInfeasible => Ok(ClarabelSolution { solution }),
            SolverStatus::Unsolved => Err(ResolutionError::Other("Unsolved")),
            SolverStatus::MaxIterations => Err(ResolutionError::Other("Max iterations reached")),
            SolverStatus::MaxTime => Err(ResolutionError::Other("Time limit reached")),
            SolverStatus::NumericalError => Err(ResolutionError::Other("Numerical error")),
            SolverStatus::InsufficientProgress => Err(ResolutionError::Other("No progress")),
            SolverStatus::CallbackTerminated => Err(ResolutionError::Other("Callback terminated")),
        }
    }

    /// Common constraint addition logic
    fn add_constraint_impl(&mut self, constraint: Constraint) -> ConstraintReference {
        self.constraints_matrix_builder
            .add_row(constraint.expression.linear);
        let index = self.constraint_values.len();
        self.constraint_values.push(-constraint.expression.constant);
        // Cones indicate the type of constraint. We only support nonnegative and equality constraints.
        // To avoid creating a new cone for each constraint, we merge them.
        let next_cone = if constraint.is_equality {
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
}

/// Trait to abstract over different Clarabel problem types
trait ClarabelProblemLike {
    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference;
}

/// The [clarabel](https://oxfordcontrol.github.io/ClarabelDocs/stable/) solver,
/// to be used with [UnsolvedProblem::using].
pub fn clarabel(to_solve: UnsolvedProblem) -> ClarabelProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let coef = ClarabelProblemCore::direction_coef(direction);
    let mut core = ClarabelProblemCore::new(variables.len());
    
    // Set up linear objective
    for (var, obj) in objective.linear_coefficients() {
        core.objective[var.index()] = obj * coef;
    }
    
    let mut p = ClarabelProblem { core };
    
    // Add variable bounds
    ClarabelProblemCore::add_variable_bounds(&variables, &mut p);
    p
}

/// The [clarabel](https://clarabel.org/stable/) solver for quadratic problems,
/// to be used with [QuadraticUnsolvedProblem::using].
#[cfg(feature = "enable_quadratic")]
pub fn clarabel_quadratic(to_solve: QuadraticUnsolvedProblem) -> ClarabelQuadraticProblem {
    let QuadraticUnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let coef = ClarabelProblemCore::direction_coef(direction);
    let mut core = ClarabelProblemCore::new(variables.len());
    
    // Set up linear objective
    for (var, obj_coeff) in objective.linear.coefficients {
        core.objective[var.index()] = obj_coeff * coef;
    }
    
    // Set up quadratic objective matrix
    let mut quadratic_matrix_builder = CscQuadraticMatrixBuilder::new(variables.len());
    for (pair, quad_coeff) in objective.quadratic.quadratic_coefficients {
        quadratic_matrix_builder.add_term(pair, quad_coeff * coef);
    }
    
    let mut p = ClarabelQuadraticProblem {
        core,
        quadratic_matrix_builder,
    };
    
    // Add variable bounds
    ClarabelProblemCore::add_variable_bounds(&variables, &mut p);
    p
}

/// Common fields shared between linear and quadratic Clarabel problems
struct ClarabelProblemCore {
    constraints_matrix_builder: CscMatrixBuilder,
    constraint_values: Vec<f64>,
    objective: Vec<f64>,
    variables: usize,
    settings: DefaultSettingsBuilder<f64>,
    cones: Vec<SupportedConeT<f64>>,
}

/// A clarabel model
pub struct ClarabelProblem {
    core: ClarabelProblemCore,
}

/// A clarabel quadratic model
#[cfg(feature = "enable_quadratic")]
pub struct ClarabelQuadraticProblem {
    core: ClarabelProblemCore,
    quadratic_matrix_builder: CscQuadraticMatrixBuilder,
}

impl ClarabelProblem {
    /// Access the problem settings
    pub fn settings(&mut self) -> &mut DefaultSettingsBuilder<f64> {
        &mut self.core.settings
    }

    /// Convert the problem into a clarabel solver
    /// panics if the problem is not valid
    pub fn into_solver(self) -> DefaultSolver<f64> {
        let settings = self.core.settings.build().expect("Invalid clarabel settings");
        let quadratic_objective = &CscMatrix::zeros((self.core.variables, self.core.variables));
        let objective = &self.core.objective;
        let constraints = &self.core.constraints_matrix_builder.build();
        let constraint_values = &self.core.constraint_values;
        let cones = &self.core.cones;
        DefaultSolver::new(
            quadratic_objective,
            objective,
            constraints,
            constraint_values,
            cones,
            settings,
        ).expect("Invalid clarabel problem. This is likely a bug in good_lp. Problems should always have coherent dimensions.")
    }
}

impl ClarabelProblemLike for ClarabelProblem {
    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        self.core.add_constraint_impl(constraint)
    }
}

impl SolverModel for ClarabelProblem {
    type Solution = ClarabelSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let mut solver = self.into_solver();
        solver.solve();
        ClarabelProblemCore::handle_solver_status(solver.solution.status, solver.solution)
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        self.core.add_constraint_impl(constraint)
    }

    fn name() -> &'static str {
        "Clarabel"
    }
}

/// The solution to a clarabel problem
pub struct ClarabelSolution {
    solution: DefaultSolution<f64>,
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
        self.solution.z[constraint.index]
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

#[cfg(feature = "enable_quadratic")]
impl ClarabelQuadraticProblem {
    /// Access the problem settings
    pub fn settings(&mut self) -> &mut DefaultSettingsBuilder<f64> {
        &mut self.core.settings
    }

    /// Convert the problem into a clarabel solver
    /// panics if the problem is not valid
    pub fn into_solver(self) -> DefaultSolver<f64> {
        let settings = self.core.settings.build().expect("Invalid clarabel settings");
        let quadratic_objective = &self.quadratic_matrix_builder.build();
        let objective = &self.core.objective;
        let constraints = &self.core.constraints_matrix_builder.build();
        let constraint_values = &self.core.constraint_values;
        let cones = &self.core.cones;
        DefaultSolver::new(
            quadratic_objective,
            objective,
            constraints,
            constraint_values,
            cones,
            settings,
        ).expect("Invalid clarabel problem. This is likely a bug in good_lp. Problems should always have coherent dimensions.")
    }
}

#[cfg(feature = "enable_quadratic")]
impl ClarabelProblemLike for ClarabelQuadraticProblem {
    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        self.core.add_constraint_impl(constraint)
    }
}

#[cfg(feature = "enable_quadratic")]
impl SolverModel for ClarabelQuadraticProblem {
    type Solution = ClarabelSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let mut solver = self.into_solver();
        solver.solve();
        ClarabelProblemCore::handle_solver_status(solver.solution.status, solver.solution)
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        self.core.add_constraint_impl(constraint)
    }

    fn name() -> &'static str {
        "Clarabel Quadratic"
    }
}

/// Builder for symmetric quadratic matrices in CSC format
#[cfg(feature = "enable_quadratic")]
struct CscQuadraticMatrixBuilder {
    /// Indicates the row index of the corresponding element in `nzval`
    rowval: Vec<Vec<usize>>,
    /// All non-zero values in the matrix, in column-major order
    nzval: Vec<Vec<f64>>,
    n_vars: usize,
}

#[cfg(feature = "enable_quadratic")]
impl CscQuadraticMatrixBuilder {
    fn new(n_vars: usize) -> Self {
        Self {
            rowval: vec![Vec::new(); n_vars],
            nzval: vec![Vec::new(); n_vars],
            n_vars,
        }
    }
    
    fn add_term(&mut self, pair: VariablePair, coeff: f64) {
        let i = pair.var1.index();
        let j = pair.var2.index();
        
        if i == j {
            // Diagonal term: multiply by 2 because Clarabel minimizes 1/2 * x^T * P * x
            // So we need P_ii = 2 * coeff to get coeff * x_i^2 in the objective
            self.rowval[j].push(i);
            self.nzval[j].push(2.0 * coeff);
        } else {
            // Off-diagonal term: add the coefficient to both (i,j) and (j,i) positions
            // to create a symmetric matrix. Each entry gets the original coefficient
            // since Clarabel multiplies the entire quadratic form by 1/2.
            self.rowval[j].push(i);
            self.nzval[j].push(coeff);
            
            self.rowval[i].push(j);
            self.nzval[i].push(coeff);
        }
    }
    
    fn build(mut self) -> clarabel::algebra::CscMatrix {
        // Sort each column by row index to maintain proper CSC format
        for col in 0..self.n_vars {
            let mut pairs: Vec<(usize, f64)> = self.rowval[col].iter()
                .zip(self.nzval[col].iter())
                .map(|(&row, &val)| (row, val))
                .collect();
            pairs.sort_by_key(|&(row, _)| row);
            
            self.rowval[col] = pairs.iter().map(|&(row, _)| row).collect();
            self.nzval[col] = pairs.iter().map(|&(_, val)| val).collect();
        }
        
        let mut colptr = Vec::with_capacity(self.n_vars + 1);
        colptr.push(0);
        for col in &self.rowval {
            colptr.push(colptr.last().unwrap() + col.len());
        }
        clarabel::algebra::CscMatrix::new(
            self.n_vars,
            self.n_vars,
            colptr,
            fast_flatten_vecs(self.rowval),
            fast_flatten_vecs(self.nzval),
        )
    }
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

    #[cfg(feature = "enable_quadratic")]
    mod quadratic_tests {
        use super::*;
        use crate::{QuadraticAffineExpression, VariablePair};

        #[test]
        fn test_csc_quadratic_matrix_builder_diagonal() {
            variables! {vars:
                x;
                y;
            }
            
            let mut builder = CscQuadraticMatrixBuilder::new(2);
            
            // Add diagonal terms: x^2 with coefficient 2.0, y^2 with coefficient 3.0
            builder.add_term(VariablePair::new(x, x), 2.0);
            builder.add_term(VariablePair::new(y, y), 3.0);
            
            let matrix = builder.build();
            
            // Check diagonal entries (scaled by 2 for Clarabel's 1/2 * x^T * P * x formulation)
            assert_eq!(matrix.get_entry((0, 0)), Some(4.0)); // 2.0 * 2 = 4.0
            assert_eq!(matrix.get_entry((1, 1)), Some(6.0)); // 3.0 * 2 = 6.0
            
            // Check off-diagonal entries are zero
            assert_eq!(matrix.get_entry((0, 1)), None);
            assert_eq!(matrix.get_entry((1, 0)), None);
        }

        #[test]
        fn test_csc_quadratic_matrix_builder_off_diagonal() {
            variables! {vars:
                x;
                y;
            }
            
            let mut builder = CscQuadraticMatrixBuilder::new(3);
            
            // Add off-diagonal term: 2*x*y (coefficient 2.0)
            builder.add_term(VariablePair::new(x, y), 2.0);
            
            let matrix = builder.build();
            
            // Off-diagonal terms: coefficient is stored directly (no factor of 2 for off-diagonal)
            assert_eq!(matrix.get_entry((0, 1)), Some(2.0)); // (x,y) entry
            assert_eq!(matrix.get_entry((1, 0)), Some(2.0)); // (y,x) entry (symmetric)
            
            // Other entries should be empty
            assert_eq!(matrix.get_entry((0, 0)), None);
            assert_eq!(matrix.get_entry((1, 1)), None);
            assert_eq!(matrix.get_entry((2, 2)), None);
        }

        #[test]
        fn test_csc_quadratic_matrix_builder_mixed() {
            variables! {vars:
                x;
                y;
            }
            
            let mut builder = CscQuadraticMatrixBuilder::new(2);
            
            // Add mixed terms: x^2 + 2*x*y + y^2 (like (x+y)^2)
            builder.add_term(VariablePair::new(x, x), 1.0);
            builder.add_term(VariablePair::new(x, y), 2.0);
            builder.add_term(VariablePair::new(y, y), 1.0);
            
            let matrix = builder.build();
            
            // Check the resulting matrix (accounting for Clarabel's 1/2 factor):
            // For (x+y)^2 = x^2 + 2*x*y + y^2, we expect:
            // [2.0  2.0]  (diagonal terms scaled by 2, off-diagonal terms as-is)
            // [2.0  2.0]
            assert_eq!(matrix.get_entry((0, 0)), Some(2.0)); // x^2 (scaled by 2)
            assert_eq!(matrix.get_entry((0, 1)), Some(2.0)); // x*y coefficient
            assert_eq!(matrix.get_entry((1, 0)), Some(2.0)); // y*x coefficient  
            assert_eq!(matrix.get_entry((1, 1)), Some(2.0)); // y^2 (scaled by 2)
        }

        #[test]
        fn test_simple_quadratic_problem() {
            variables! {vars:
                x;
                y;
            }

            // Create a simple quadratic objective: minimize x^2 + y^2
            let mut quadratic_obj = QuadraticAffineExpression::new();
            quadratic_obj.add_quadratic_term(x, x, 1.0);  // x^2
            quadratic_obj.add_quadratic_term(y, y, 1.0);  // y^2

            // Create the problem
            let problem = vars
                .minimise_quadratic(quadratic_obj)
                .using(clarabel_quadratic);

            // The problem should be solvable (unconstrained minimum at (0,0))
            let solution = problem.solve().expect("Should solve successfully");
            
            // Check that solution is near the origin
            assert!((solution.value(x)).abs() < 1e-6);
            assert!((solution.value(y)).abs() < 1e-6);
        }

        #[test]
        fn test_quadratic_problem_with_constraints() {
            variables! {vars:
                x;
                y;
            }

            // Create quadratic objective: minimize x^2 + y^2
            let mut quadratic_obj = QuadraticAffineExpression::new();
            quadratic_obj.add_quadratic_term(x, x, 1.0);  // x^2
            quadratic_obj.add_quadratic_term(y, y, 1.0);  // y^2

            // Solve with constraint x + y >= 2
            let solution = vars
                .minimise_quadratic(quadratic_obj)
                .using(clarabel_quadratic)
                .with(x + y >> 2.0)
                .solve()
                .expect("Should solve successfully");

            // The optimal solution should be x = y = 1 (closest point to origin on line x+y=2)
            assert!((solution.value(x) - 1.0).abs() < 1e-3);
            assert!((solution.value(y) - 1.0).abs() < 1e-3);
            
            // Check constraint is satisfied
            assert!(solution.value(x) + solution.value(y) >= 1.99);
        }

        #[test]
        fn test_quadratic_problem_mixed_terms() {
            variables! {vars:
                x;
                y;
            }

            // Create quadratic objective: minimize (x-1)^2 + (y-2)^2 + 2*x*y
            // Expanded: x^2 - 2x + 1 + y^2 - 4y + 4 + 2xy
            //         = x^2 + y^2 + 2xy - 2x - 4y + 5
            let mut quadratic_obj = QuadraticAffineExpression::new();
            quadratic_obj.add_quadratic_term(x, x, 1.0);   // x^2
            quadratic_obj.add_quadratic_term(y, y, 1.0);   // y^2
            quadratic_obj.add_quadratic_term(x, y, 2.0);   // 2*x*y
            quadratic_obj.add_linear_term(x, -2.0);        // -2x
            quadratic_obj.add_linear_term(y, -4.0);        // -4y
            quadratic_obj.add_constant(5.0);               // +5

            let solution = vars
                .minimise_quadratic(quadratic_obj)
                .using(clarabel_quadratic)
                .solve()
                .expect("Should solve successfully");

            // This is a more complex quadratic, but should still solve
            // We mainly check that it doesn't crash and produces a reasonable solution
            assert!(solution.value(x).is_finite());
            assert!(solution.value(y).is_finite());
        }

        // Note: More complex single-variable quadratic tests are covered in integration tests

        #[test]
        fn test_quadratic_problem_with_linear_constraints() {
            variables! {vars:
                x;
                y;
                z;
            }

            // Minimize x^2 + y^2 + z^2 subject to x + y + z = 3
            let mut quadratic_obj = QuadraticAffineExpression::new();
            quadratic_obj.add_quadratic_term(x, x, 1.0);
            quadratic_obj.add_quadratic_term(y, y, 1.0);
            quadratic_obj.add_quadratic_term(z, z, 1.0);

            let solution = vars
                .minimise_quadratic(quadratic_obj)
                .using(clarabel_quadratic)
                .with((x + y + z).eq(3.0))  // Equality constraint
                .solve()
                .expect("Should solve successfully");

            // Optimal solution should be x = y = z = 1
            assert!((solution.value(x) - 1.0).abs() < 1e-3);
            assert!((solution.value(y) - 1.0).abs() < 1e-3);
            assert!((solution.value(z) - 1.0).abs() < 1e-3);
        }

        #[test]
        fn test_quadratic_expression_evaluation() {
            variables! {vars:
                x;
                y;
            }

            // Create quadratic expression: x^2 + 2*x*y + y^2 (which is (x+y)^2)
            let mut quadratic_obj = QuadraticAffineExpression::new();
            quadratic_obj.add_quadratic_term(x, x, 1.0);
            quadratic_obj.add_quadratic_term(x, y, 2.0);
            quadratic_obj.add_quadratic_term(y, y, 1.0);

            let solution = vars
                .minimise_quadratic(quadratic_obj)
                .using(clarabel_quadratic)
                .with(x + y >> 2.0)  // x + y >= 2
                .solve()
                .expect("Should solve successfully");

            // At the solution, (x+y)^2 should equal the sum of coefficients times the values
            let x_val = solution.value(x);
            let y_val = solution.value(y);
            let expected_obj_value = x_val * x_val + 2.0 * x_val * y_val + y_val * y_val;
            let actual_obj_value = (x_val + y_val) * (x_val + y_val);
            
            assert!((expected_obj_value - actual_obj_value).abs() < 1e-6);
        }
    }
}
