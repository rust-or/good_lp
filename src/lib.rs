#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//!  A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.
//!
//!  ```rust
//!  use good_lp::{variables, variable, default_solver, SolverModel, Solution};
//!
//! // Create variables in a readable format with a macro...
//! variables!{
//!     vars:
//!         a <= 1;
//!         2 <= b <= 4;
//! }
//! // ... or add variables programmatically
//! vars.add(variable().min(2).max(9));
//!
//!  let solution = vars.maximise(10 * (a - b / 5) - b)
//!      .using(default_solver)
//!      .with(a + 2. << b) // or (a + 2).leq(b)
//!      .with(1 + a >> 4. - b)
//!      .solve()?;
//!
//! # use float_eq::assert_float_eq;
//!  assert_float_eq!(solution.value(a), 1., abs <= 1e-8);
//!  assert_float_eq!(solution.value(b), 3., abs <= 1e-8);
//!  # Ok::<_, good_lp::ResolutionError>(())
//!  ```
//!
//! ## Solvers
//!
//! This crate supports multiple solvers,
//! that can be activated using [feature flags](https://docs.rs/crate/good_lp/latest/features).
//!
//! ## Usage
//!
//! You initially create your variables using [variables] and [ProblemVariables::add].
//!
//! Then you create your objective function.If it's large, you can write rust functions
//! to split complex expressions into components.
//!
//! ```
//! use good_lp::{Expression, Variable};
//!
//! fn total_cost(energy: Variable, time: Variable) -> Expression {
//! #   let dollars_per_hour = 0;
//!     energy_cost(energy) + dollars_per_hour * time
//! }
//!
//! fn energy_cost(energy: Variable) -> Expression {
//! #   let fetch_energy_price = |_| 0.;
//!     let price = fetch_energy_price(energy);
//!     energy * price
//! }
//! ```
//!
//! Then you create a [solver](solvers) problem model instance
//! ```
//! # let my_variables = good_lp::variables!();
//! # let my_objective = good_lp::Expression::from(0);
//! # let my_solver = good_lp::default_solver;
//! let mut model = my_variables.minimise(my_objective).using(my_solver);
//! ```
//!
//! Then you add constraints and solve your problem using the methods in [SolverModel].
//!

pub use affine_expression_trait::IntoAffineExpression;
pub use cardinality_constraint_solver_trait::CardinalityConstraintSolver;
pub use constraint::Constraint;
pub use expression::Expression;
#[cfg(not(any(
    feature = "coin_cbc",
    feature = "microlp",
    feature = "lpsolve",
    feature = "highs",
    feature = "scip",
    feature = "cplex-rs",
)))]
#[cfg(feature = "clarabel")]
pub use solvers::clarabel::clarabel as default_solver;
#[cfg_attr(docsrs, doc(cfg(feature = "clarabel")))]
#[cfg(feature = "clarabel")]
pub use solvers::clarabel::clarabel;
#[cfg_attr(docsrs, doc(cfg(feature = "coin_cbc")))]
#[cfg(feature = "coin_cbc")]
pub use solvers::coin_cbc::coin_cbc;
#[cfg(feature = "coin_cbc")]
/// When the "coin_cbc" cargo feature is present, it is used as the default solver
pub use solvers::coin_cbc::coin_cbc as default_solver;
#[cfg(not(any(
    feature = "coin_cbc",
    feature = "microlp",
    feature = "lpsolve",
    feature = "highs"
)))]
#[cfg(feature = "cplex-rs")]
pub use solvers::cplex::cplex as default_solver;
#[cfg(feature = "highs")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub use solvers::highs::highs;
#[cfg(not(any(feature = "coin_cbc", feature = "microlp", feature = "lpsolve")))]
#[cfg(feature = "highs")]
/// When the "highs" cargo feature is present, highs is used as the default solver
pub use solvers::highs::highs as default_solver;
#[cfg(feature = "lp-solvers")]
#[cfg_attr(docsrs, doc(cfg(feature = "lp-solvers")))]
pub use solvers::lp_solvers::LpSolver;
#[cfg(feature = "lpsolve")]
#[cfg_attr(docsrs, doc(cfg(feature = "lpsolve")))]
pub use solvers::lpsolve::lp_solve;
#[cfg(not(any(feature = "coin_cbc", feature = "microlp")))]
#[cfg(feature = "lpsolve")]
/// When the "lpsolve" cargo feature is present, lpsolve is used as the default solver
pub use solvers::lpsolve::lp_solve as default_solver;
#[cfg(feature = "microlp")]
#[cfg_attr(docsrs, doc(cfg(feature = "microlp")))]
pub use solvers::microlp::microlp;
#[cfg(not(feature = "coin_cbc"))]
#[cfg(feature = "microlp")]
/// When the "coin_cbc" cargo feature is absent, microlp is used as the default solver
pub use solvers::microlp::microlp as default_solver;
#[cfg(feature = "scip")]
#[cfg_attr(docsrs, doc(cfg(feature = "highs")))]
pub use solvers::scip::scip;
#[cfg(not(any(
    feature = "coin_cbc",
    feature = "microlp",
    feature = "lpsolve",
    feature = "highs"
)))]
#[cfg(feature = "scip")]
pub use solvers::scip::scip as default_solver;

pub use solvers::{
    solver_name, DualValues, ModelWithSOS1, ResolutionError, Solution, SolutionStatus,
    SolutionWithDual, Solver, SolverModel, StaticSolver, WithInitialSolution, WithMipGap,
    WithTimeLimit,
};
pub use variable::{variable, ProblemVariables, Variable, VariableDefinition};

#[cfg(not(any(
    feature = "coin_cbc",
    feature = "microlp",
    feature = "lpsolve",
    feature = "highs",
    feature = "scip",
)))]
#[cfg(feature = "lp-solvers")]
/// Default solvers for the 'lp-solvers' feature: a solver that calls Cbc as an external command
#[allow(non_upper_case_globals)]
pub const default_solver: LpSolver<
    solvers::lp_solvers::StaticSolver<solvers::lp_solvers::AllSolvers>,
> = LpSolver(solvers::lp_solvers::StaticSolver::new());

#[cfg(not(any(
    feature = "coin_cbc",
    feature = "microlp",
    feature = "lpsolve",
    feature = "highs",
    feature = "lp-solvers",
    feature = "scip",
    feature = "cplex-rs",
    feature = "clarabel",
)))]
compile_error!(
    "No solver available. \
You need to activate at least one solver feature flag in good_lp. \
You can do by adding the following to your Cargo.toml :
[dependencies]
good_lp = { version = \"*\", features = [\"microlp\"] }
"
);

/// The "lpsolve" and "cplex-rs" cargo feature are incompatible,
/// since the above crates link statically to c libraries defining some
/// common symbols
#[cfg(all(feature = "lpsolve", feature = "cplex-rs",))]
compile_error!(
    "'lpsolve' and 'cplex-rs' features are incompatible. \
Please select just one of the two. If you need all compatible solvers, use the 'all_deafult_solvers' feature. \
"
);

mod expression;
#[macro_use]
pub mod variable;
mod affine_expression_trait;
mod cardinality_constraint_solver_trait;
pub mod constraint;
pub mod solvers;
mod variables_macro;
