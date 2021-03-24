use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use fnv::FnvHashMap as HashMap;

use crate::affine_expression_trait::IntoAffineExpression;
use crate::constraint;
use crate::variable::{FormatWithVars, Variable};
use crate::{Constraint, Solution};

/// An linear expression without a constant component
pub struct LinearExpression {
    pub(crate) coefficients: HashMap<Variable, f64>,
}

impl IntoAffineExpression for LinearExpression {
    type Iter = std::collections::hash_map::IntoIter<Variable, f64>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.coefficients.into_iter()
    }
}

/// Return type for `&'a LinearExpression::linear_coefficients`
#[doc(hidden)]
pub struct CopiedCoefficients<'a>(std::collections::hash_map::Iter<'a, Variable, f64>);

impl<'a> Iterator for CopiedCoefficients<'a> {
    type Item = (Variable, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(&var, &c)| (var, c))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a> IntoAffineExpression for &'a LinearExpression {
    type Iter = CopiedCoefficients<'a>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        CopiedCoefficients(self.coefficients.iter())
    }
}

impl FormatWithVars for LinearExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let mut first = true;
        for (&var, &coeff) in &self.coefficients {
            if coeff != 0f64 {
                if first {
                    first = false;
                } else {
                    write!(f, " + ")?;
                }
                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, var)?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

/// Represents an affine expression, such as `2x + 3` or `x + y + z`
pub struct Expression {
    pub(crate) linear: LinearExpression,
    pub(crate) constant: f64,
}

impl IntoAffineExpression for Expression {
    type Iter = <LinearExpression as IntoAffineExpression>::Iter;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.linear.linear_coefficients()
    }

    #[inline]
    fn constant(&self) -> f64 {
        self.constant
    }
}

/// This implementation copies all the variables and coefficients from the referenced
/// Expression into the created iterator
impl<'a> IntoAffineExpression for &'a Expression {
    type Iter = <&'a LinearExpression as IntoAffineExpression>::Iter;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        (&self.linear).linear_coefficients()
    }

    #[inline]
    fn constant(&self) -> f64 {
        self.constant
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.constant.eq(&other.constant) && self.linear.coefficients.eq(&other.linear.coefficients)
    }
}

impl Clone for Expression {
    fn clone(&self) -> Self {
        Expression {
            linear: LinearExpression {
                coefficients: self.linear.coefficients.clone(),
            },
            constant: self.constant,
        }
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::from(0.)
    }
}

impl Expression {
    /// Create an expression that has the value 0, but has memory allocated
    /// for `capacity` coefficients.
    pub fn with_capacity(capacity: usize) -> Self {
        Expression {
            linear: LinearExpression {
                coefficients: HashMap::with_capacity_and_hasher(capacity, Default::default()),
            },
            constant: 0.0,
        }
    }

    /// Create a concrete expression struct from anything that has linear coefficients and a constant
    ///
    /// ```
    /// # use good_lp::Expression;
    /// Expression::from_other_affine(0.); // A constant expression
    /// ```
    pub fn from_other_affine<E: IntoAffineExpression>(source: E) -> Self {
        source.into_expression()
    }

    /// Creates a constraint indicating that this expression
    /// is lesser than or equal to the right hand side
    pub fn leq<RHS>(self, rhs: RHS) -> Constraint
    where
        Expression: Sub<RHS, Output = Expression>,
    {
        constraint::leq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is greater than or equal to the right hand side
    pub fn geq<RHS: Sub<Expression, Output = Expression>>(self, rhs: RHS) -> Constraint {
        constraint::geq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is equal to the right hand side
    pub fn eq<RHS>(self, rhs: RHS) -> Constraint
    where
        Expression: Sub<RHS, Output = Expression>,
    {
        constraint::eq(self, rhs)
    }

    /// Performs self = self + (a * b)
    #[inline]
    pub fn add_mul<N: Into<f64>, E: IntoAffineExpression>(&mut self, a: N, b: E) {
        let factor = a.into();
        let constant = b.constant();
        for (var, value) in b.linear_coefficients().into_iter() {
            *self.linear.coefficients.entry(var).or_default() += factor * value
        }
        self.constant += factor * constant;
    }

    /// See [IntoAffineExpression::eval_with]
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        IntoAffineExpression::eval_with(self, values)
    }
}

#[inline]
pub fn add_mul<LHS: Into<Expression>, RHS: IntoAffineExpression>(
    lhs: LHS,
    rhs: RHS,
    factor: f64,
) -> Expression {
    let mut result = lhs.into();
    result.add_mul(factor, rhs);
    result
}

#[inline]
pub fn sub<LHS: Into<Expression>, RHS: IntoAffineExpression>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, -1.)
}

#[inline]
pub fn add<LHS: Into<Expression>, RHS: IntoAffineExpression>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, 1.)
}

impl FormatWithVars for Expression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        self.linear.format_with(f, variable_format)?;
        if self.constant.abs() >= f64::EPSILON {
            write!(f, " + {}", self.constant)?;
        }
        Ok(())
    }
}

impl<RHS: IntoAffineExpression> SubAssign<RHS> for Expression {
    #[inline]
    fn sub_assign(&mut self, rhs: RHS) {
        self.add_mul(-1., rhs)
    }
}

impl<RHS: IntoAffineExpression> AddAssign<RHS> for Expression {
    #[inline]
    fn add_assign(&mut self, rhs: RHS) {
        self.add_mul(1, rhs);
    }
}

impl Neg for Expression {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

impl<N: Into<f64>> MulAssign<N> for Expression {
    #[inline]
    fn mul_assign(&mut self, rhs: N) {
        let factor = rhs.into();
        for value in self.linear.coefficients.values_mut() {
            *value *= factor
        }
        self.constant *= factor
    }
}

impl<N: Into<f64>> Mul<N> for Expression {
    type Output = Expression;

    #[inline]
    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<N: Into<f64>> Div<N> for Expression {
    type Output = Expression;

    #[inline]
    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

macro_rules! impl_mul {
    ($($t:ty),*) =>{$(
        impl Mul<Expression> for $t {
            type Output = Expression;

            fn mul(self, mut rhs: Expression) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    )*}
}
impl_mul!(f64, i32);

macro_rules! impl_ops_local {
    ($( $typename:ident : $([generic $generic:ident])? $other:ident),*) => {$(
        impl<$($generic: IntoAffineExpression)?> Sub<$other>
        for $typename {
            type Output = Expression;
            fn sub(self, rhs: $other) -> Self::Output { sub(self, rhs) }
        }

        impl<$($generic: IntoAffineExpression)?> Add<$other>
        for $typename {
            type Output = Expression;
            fn add(self, rhs: $other) -> Self::Output { add(self, rhs) }
        }
    )*}
}

impl_ops_local!(
    Expression: [generic RHS] RHS,
    Variable: [generic RHS] RHS,
    f64: Expression,
    f64: Variable,
    i32: Expression,
    i32: Variable
);

macro_rules! impl_conv {
    ( $( $typename:ident ),* ) => {$(
        impl From<$typename> for Expression {
            fn from(x: $typename) -> Expression { Expression::from_other_affine(x) }
        }
    )*}
}
impl_conv!(f64, i32, Variable);

impl<E: IntoAffineExpression> std::iter::Sum<E> for Expression {
    fn sum<I: Iterator<Item = E>>(iter: I) -> Self {
        let (capacity, _) = iter.size_hint();
        let mut res = Expression::with_capacity(capacity);
        for i in iter {
            res.add_assign(i)
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::variables;

    #[test]
    fn expression_manipulation() {
        variables! {vars: v0; v1; }
        assert_eq!((3. - v0) - v1, (-1.) * v0 + (-1.) * v1 + 3.)
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn eval() {
        let mut vars = variables!();
        let a = vars.add_variable();
        let b = vars.add_variable();
        let mut values = HashMap::new();
        values.insert(a, 100);
        values.insert(b, -1);
        assert_eq!((a + 3 * (b + 3)).eval_with(&values), 106.)
    }
}
