#![deny(missing_docs)]
//!  A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.
//!
//!  ```rust
//! #[cfg(feature = "coin_cbc")] {
//!  use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
//!
//!  let mut vars = variables!();
//!  let a = vars.add(variable().max(1));
//!  let b = vars.add(variable().min(2).max(4));
//!  let solution = vars.maximise(10 * (a - b / 5) - b)
//!      .using(coin_cbc)
//!      .with(a + 2. << b)
//!      .with(1 + a >> 4. - b)
//!      .solve()?;
//!
//!  assert_eq!(solution.value(a), 1.);
//!  assert_eq!(solution.value(b), 3.);
//!  # } Ok::<_, good_lp::ResolutionError>(())
//!  ```

pub use constraint::Constraint;
pub use expression::Expression;
pub use solvers::{ResolutionError, Solution, SolverModel};
pub use variable::{variable, Variable};

#[cfg(feature = "coin_cbc")]
pub use solvers::coin_cbc::coin_cbc;

#[cfg(feature = "minilp")]
pub use solvers::minilp::minilp;

mod expression;
#[macro_use]
pub mod variable;
pub mod constraint;
mod solvers;
mod variables_macro;
