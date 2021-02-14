#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
//!  A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.
//!
//!  ```rust
//! # #[cfg(feature = "coin_cbc")] {
//!  use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
//!
//!  let mut vars = variables!();
//!  let a = vars.add(variable().max(1));
//!  let b = vars.add(variable().min(2).max(4));
//!  let solution = vars.maximise(10 * (a - b / 5) - b)
//!      .using(coin_cbc)
//!      .with(a + 2. << b) // or (a + 2).leq(b)
//!      .with(1 + a >> 4. - b)
//!      .solve()?;
//!
//!  assert_eq!(solution.value(a), 1.);
//!  assert_eq!(solution.value(b), 3.);
//!  # } Ok::<_, good_lp::ResolutionError>(())
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
//! fn total_cost<V>(energy: Variable<V>, time: Variable<V>) -> Expression<V> {
//! #   let dollars_per_hour = 0;
//!     energy_cost(energy) + dollars_per_hour * time
//! }
//!
//! fn energy_cost<V>(energy: Variable<V>) -> Expression<V> {
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
//! # let my_solver = |_|();
//! let mut model = my_variables.minimise(my_objective).using(my_solver);
//! ```
//!
//! Then you add constraints and solve your problem using the methods in [SolverModel].
//!

pub use constraint::Constraint;
pub use expression::Expression;
pub use solvers::{ResolutionError, Solution, SolverModel};
pub use variable::{variable, ProblemVariables, Variable, VariableDefinition};

#[cfg(feature = "coin_cbc")]
#[cfg_attr(docsrs, doc(cfg(feature = "minilp")))]
pub use solvers::coin_cbc::coin_cbc;

#[cfg(feature = "minilp")]
#[cfg_attr(docsrs, doc(cfg(feature = "minilp")))]
pub use solvers::minilp::minilp;

mod expression;
#[macro_use]
pub mod variable;
pub mod constraint;
pub mod solvers;
mod variables_macro;
