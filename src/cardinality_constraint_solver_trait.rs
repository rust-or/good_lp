use crate::{Variable, constraint::ConstraintReference};

/// A trait for solvers that support cardinality constraints
pub trait CardinalityConstraintSolver {
    /// Add cardinality constraint. Constrains the number of non-zero variables from `vars` to at most `rhs`.
    fn add_cardinality_constraint(&mut self, vars: &[Variable], rhs: usize) -> ConstraintReference;
}
