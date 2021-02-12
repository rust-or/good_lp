//! Constraints define the inequalities that must hold in the solution.
use crate::expression::Expression;
use crate::variable::{FormatWithVars, Variable};
use core::fmt::{Debug, Formatter};
use std::ops::{Shl, Shr};

/// A constraint represents a single (in)equality that must hold in the solution.
pub struct Constraint<F> {
    pub(crate) expression: Expression<F>,
    /// if is_equality, represents expression == 0, otherwise, expression <= 0
    pub(crate) is_equality: bool,
}

impl<F> Constraint<F> {
    fn new(expression: Expression<F>, is_equality: bool) -> Constraint<F> {
        Constraint {
            expression,
            is_equality,
        }
    }
}

impl<F> FormatWithVars<F> for Constraint<F> {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: Fn(&mut Formatter<'_>, Variable<F>) -> std::fmt::Result,
    {
        self.expression.linear.format_with(f, variable_format)?;
        write!(f, " {} ", if self.is_equality { "=" } else { "<=" })?;
        write!(f, "{}", -self.expression.constant)
    }
}

impl<F> Debug for Constraint<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

/// equals
pub fn eq<F, A: Into<Expression<F>>, B: Into<Expression<F>>>(a: A, b: B) -> Constraint<F> {
    Constraint::new(a.into() - b.into(), true)
}

/// less than or equal
pub fn leq<F, A: Into<Expression<F>>, B: Into<Expression<F>>>(a: A, b: B) -> Constraint<F> {
    Constraint::new(a.into() - b.into(), false)
}

/// greater than or equal
pub fn geq<F, A: Into<Expression<F>>, B: Into<Expression<F>>>(a: A, b: B) -> Constraint<F> {
    leq(b, a)
}

macro_rules! impl_shifts {
    ($($t:ty)*) => {$(
        impl<F, RHS: Into<Expression<F>>> Shl<RHS> for $t {
            type Output = Constraint<F>;

            fn shl(self, rhs: RHS) -> Self::Output {
                leq(self, rhs)
            }
        }

        impl<F, RHS: Into<Expression<F>>> Shr<RHS> for $t {
            type Output = Constraint<F>;

            fn shr(self, rhs: RHS) -> Self::Output {
                geq(self, rhs)
            }
        }
    )*}
}

impl_shifts!(Expression<F> Variable<F>);

#[cfg(test)]
mod tests {
    use crate::variables;
    #[test]
    fn test_leq() {
        let mut vars = variables!();
        let v0 = vars.add_variable();
        let v1 = vars.add_variable();
        let f = format!("{:?}", (3. - v0) >> v1);
        assert!(vec!["v0 + v1 <= 3", "v1 + v0 <= 3"].contains(&&*f), "{}", f)
    }
}
