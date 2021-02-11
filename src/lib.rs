pub use expression::Expression;
pub use variable::{Variable, Constraint};

mod expression;
#[macro_use]
pub mod variable;
mod solvers;