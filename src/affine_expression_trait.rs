//! An affine expression is an expression of the form `a + 2b - 3a - 7`
//! You can implement this trait if you want to implement your own
//! variant of the [Expression](crate::Expression) type, optimized for your use case.
use crate::expression::LinearExpression;
use crate::{Expression, Variable};

/// An element that can be expressed as a linear combination of variables plus a constant
pub trait IntoAffineExpression {
    /// The iterator returned by [`linear_coefficients`](IntoAffineExpression::linear_coefficients).
    type Iter: IntoIterator<Item = (Variable, f64)>;

    /// An iterator over variables and their coefficients.
    /// For instance `a + 2b - 3a - 7` should yield `[(a, -2), (b, 2)]`
    fn linear_coefficients(self) -> Self::Iter;

    /// The constant factor in the expression.
    /// For instance, `a + 2b - 7` will give `-7`
    #[inline]
    fn constant(&self) -> f64 {
        0.
    }

    /// Transform the value into a concrete Expression struct.
    fn into_expression(self) -> Expression
    where
        Self: Sized,
    {
        let constant = self.constant();
        let coefficients = self.linear_coefficients().into_iter().collect();
        Expression {
            linear: LinearExpression { coefficients },
            constant,
        }
    }
}

macro_rules! impl_affine_for_num {
    ($($num:ty),*) => {$(
        impl IntoAffineExpression for $num {
            type Iter = std::iter::Empty<(Variable, f64)>;

            #[inline]
            fn linear_coefficients(self) -> Self::Iter {
                std::iter::empty()
            }

            #[inline]
            fn constant(&self) -> f64 {
                f64::from(*self)
            }

            fn into_expression(self) -> Expression {
                Expression {
                    linear: LinearExpression { coefficients: std::default::Default::default() },
                    constant: f64::from(self),
                }
            }
        }
    )*};
}

impl_affine_for_num!(f64, f32, u32, u16, u8, i32, i16, i8);
