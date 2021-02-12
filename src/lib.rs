/// A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.
///
/// ```rust
/// use good_lp::{variables, coin_cbc, SolverModel, Solution};
///
/// let mut vars = variables!();
/// let a = vars.add_variable();
/// let b = vars.add_variable();
/// let solution = vars.maximise(9. * (a * 2 + b / 3))
///     .using(coin_cbc)
///     .with(a + 2. << b)
///     .with(3. - a >> b)
///     .solve()?;
///
/// assert_eq!(solution.value(a), 0.5);
/// assert_eq!(solution.value(b), 2.5);
/// # use good_lp::ResolutionError;
/// # Ok::<_, ResolutionError>(())
/// ```

pub use expression::Expression;
pub use variable::Variable;
pub use constraint::Constraint;
pub use solvers::{ResolutionError, Solution, SolverModel};
pub use solvers::coin_cbc::coin_cbc;

mod expression;
#[macro_use]
pub mod variable;
mod solvers;
pub mod constraint;