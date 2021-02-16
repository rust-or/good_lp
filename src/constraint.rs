//! Constraints define the inequalities that must hold in the solution.
use crate::expression::Expression;
use crate::variable::{FormatWithVars, Variable};
use core::fmt::{Debug, Formatter};
use std::ops::{Shl, Shr, Sub};

/// A constraint represents a single (in)equality that must hold in the solution.
pub struct Constraint {
    /// The expression that is constrained to be null or negative
    pub(crate) expression: Expression,
    /// if is_equality, represents expression == 0, otherwise, expression <= 0
    pub(crate) is_equality: bool,
}

impl Constraint {
    fn new(expression: Expression, is_equality: bool) -> Constraint {
        Constraint {
            expression,
            is_equality,
        }
    }
}

impl FormatWithVars for Constraint {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: Fn(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        self.expression.linear.format_with(f, variable_format)?;
        write!(f, " {} ", if self.is_equality { "=" } else { "<=" })?;
        write!(f, "{}", -self.expression.constant)
    }
}

impl Debug for Constraint {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

/// equals
pub fn eq<B, A: Sub<B, Output = Expression>>(a: A, b: B) -> Constraint {
    Constraint::new(a - b, true)
}

/// less than or equal
pub fn leq<B, A: Sub<B, Output = Expression>>(a: A, b: B) -> Constraint {
    Constraint::new(a - b, false)
}

/// greater than or equal
pub fn geq<A, B: Sub<A, Output = Expression>>(a: A, b: B) -> Constraint {
    leq(b, a)
}

macro_rules! impl_shifts {
    ($($t:ty)*) => {$(
        impl< RHS> Shl<RHS> for $t where Self: Sub<RHS, Output=Expression> {
            type Output = Constraint;

            fn shl(self, rhs: RHS) -> Self::Output {
                leq(self, rhs)
            }
        }

        impl< RHS: Sub<Self, Output=Expression>> Shr<RHS> for $t {
            type Output = Constraint;

            fn shr(self, rhs: RHS) -> Self::Output {
                geq(self, rhs)
            }
        }
    )*}
}

impl_shifts!(Expression Variable);

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
