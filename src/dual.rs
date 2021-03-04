//! The dual value measures the increase in the objective function's value per unit
//! increase in the variable's value. The dual value for a constraint is nonzero only when
//! the constraint is equal to its bound. Also known as the shadow price
//!

/// Interface to handle the retrieval of Dual Values from a solver
pub trait Dual<'a> {
    /// Extract function to get values
    fn get_dual(&'a mut self) -> &'a Self;
}
