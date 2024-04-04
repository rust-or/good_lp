//! An affine expression is an expression of the form `a + 2b - 3a - 7`
//! You can implement this trait if you want to implement your own
//! variant of the [Expression](crate::Expression) type, optimized for your use case.
use crate::expression::LinearExpression;
use crate::{Expression, Solution, Variable};

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

    /// Evaluate the concrete value of the expression, given the values of the variables
    ///
    /// ## Examples
    /// ### Evaluate an expression using a solution
    ///
    /// ```rust
    /// use good_lp::{variables, variable, default_solver, SolverModel, Solution};
    /// variables!{ vars: a <= 1; b <= 4; }
    /// let objective = a + b;
    /// let solution = vars.maximise(objective.clone()).using(default_solver).solve()?;
    /// assert_eq!(objective.eval_with(&solution), 5.);
    /// # use good_lp::ResolutionError;
    /// # Ok::<_, ResolutionError>(())
    /// ```
    ///
    /// ### Evaluate an expression with a HashMap
    /// A [std::collections::HashMap] is a valid [Solution]
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use good_lp::{variables, Variable};
    /// let mut vars = variables!();
    /// let a = vars.add_variable();
    /// let b = vars.add_variable();
    /// let expr = a + b / 2;
    /// let var_mapping: HashMap<_, _> = vec![(a, 3), (b, 10)].into_iter().collect();
    /// let value = expr.eval_with(&var_mapping);
    /// assert_eq!(value, 8.);
    /// ```
    fn eval_with<S: Solution>(self, values: &S) -> f64
    where
        Self: Sized,
    {
        self.constant()
            + self
                .linear_coefficients()
                .into_iter()
                .map(|(var, coefficient)| coefficient * values.value(var))
                .sum::<f64>()
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
