//! The dual value measures the increase in the objective function's value per unit
//! increase in the variable's value. The dual value for a constraint is nonzero only when
//! the constraint is equal to its bound. Also known as the shadow price
//!

/// A problem dual values (shadow prices)
pub trait DualValues {
    /// Get all dual values for a given problem
    fn get_dual_values(&self) -> &[f64];
    /// Get the dual value of a problem by index
    fn get_dual_value(&self, constraint_index: usize) -> f64;
}
