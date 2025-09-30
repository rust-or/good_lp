//! A quadratic expression trait for expressions that can be converted to quadratic form.
//! A quadratic expression contains linear terms, quadratic terms, and constants.

#[cfg(feature = "enable_quadratic")]
use crate::{Expression, Variable};

/// An element that can be expressed as a quadratic expression
/// (combination of quadratic terms, linear terms, and constants)
#[cfg(feature = "enable_quadratic")]
pub trait IntoQuadraticExpression {
    /// Convert this element into a quadratic expression
    fn into_quadratic_expression(self) -> Expression
    where
        Self: Sized;
}

// Implement the trait for Expression (which can now contain quadratic terms)
#[cfg(feature = "enable_quadratic")]
impl IntoQuadraticExpression for Expression {
    fn into_quadratic_expression(self) -> Expression {
        self
    }
}

// Implement the trait for variables
#[cfg(feature = "enable_quadratic")]
impl IntoQuadraticExpression for Variable {
    fn into_quadratic_expression(self) -> Expression {
        Expression::from(self)
    }
}

// Implement the trait for numeric constants
#[cfg(feature = "enable_quadratic")]
impl IntoQuadraticExpression for f64 {
    fn into_quadratic_expression(self) -> Expression {
        Expression::from(self)
    }
}

#[cfg(feature = "enable_quadratic")]
impl IntoQuadraticExpression for i32 {
    fn into_quadratic_expression(self) -> Expression {
        (self as f64).into_quadratic_expression()
    }
}

// Blanket implementation for references
#[cfg(feature = "enable_quadratic")]
impl<T: Clone + IntoQuadraticExpression> IntoQuadraticExpression for &T {
    fn into_quadratic_expression(self) -> Expression {
        self.clone().into_quadratic_expression()
    }
}
