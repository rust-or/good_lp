//! Constraints define the inequalities that must hold in the solution.
use crate::expression::Expression;
use crate::variable::{FormatWithVars, Variable};
use core::fmt::{Debug, Formatter};
use std::ops::{Shl, Shr, Sub};

/// A constraint represents a single (in)equality that must hold in the solution.
#[derive(Clone)]
pub struct Constraint {
    /// The expression that is constrained to be null or negative
    pub(crate) expression: Expression,
    /// The direction of the constraint before it is normalized for a solver.
    pub(crate) direction: ConstraintDirection,
    /// Optional constraint name
    pub(crate) name: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConstraintDirection {
    LessOrEqual,
    Equal,
    GreaterOrEqual,
}

impl ConstraintDirection {
    fn is_equality(self) -> bool {
        matches!(self, Self::Equal)
    }

    #[cfg(any(feature = "highs", feature = "clarabel"))]
    fn dual_sign(self) -> f64 {
        // `>=` constraints are normalized by reversing the expression and its RHS.
        if matches!(self, Self::GreaterOrEqual) {
            -1.
        } else {
            1.
        }
    }
}

impl Constraint {
    fn new(expression: Expression, direction: ConstraintDirection) -> Constraint {
        Constraint {
            expression,
            direction,
            name: None,
        }
    }

    /// set the constraint name
    pub fn set_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// The expression that is constrained to be null or negative
    pub fn expression(&self) -> &Expression {
        &self.expression
    }

    /// Returns whether this constraint is an equality.
    pub fn is_equality(&self) -> bool {
        self.direction.is_equality()
    }

    pub(crate) fn reference(&self, index: usize) -> ConstraintReference {
        ConstraintReference::with_direction(index, self.direction)
    }

    /// get the constraint name, if it exists.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl FormatWithVars for Constraint {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        self.expression.linear.format_with(f, variable_format)?;
        write!(
            f,
            " {} ",
            if self.direction.is_equality() {
                "="
            } else {
                "<="
            }
        )?;
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
    Constraint::new(a - b, ConstraintDirection::Equal)
}

/// less than or equal
pub fn leq<B, A: Sub<B, Output = Expression>>(a: A, b: B) -> Constraint {
    Constraint::new(a - b, ConstraintDirection::LessOrEqual)
}

/// greater than or equal
pub fn geq<A, B: Sub<A, Output = Expression>>(a: A, b: B) -> Constraint {
    // Keep the original direction so dual values can be mapped back after normalization.
    Constraint::new(b - a, ConstraintDirection::GreaterOrEqual)
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

/// This macro allows defining constraints using `a + b <= c + d`
/// instead of `(a + b).leq(c + d)` or `a + b << c + d`
///
/// # Example
///
/// ## Create a constraint
///
/// ```
/// # use good_lp::*;
/// # let mut vars = variables!();
/// # let a = vars.add(variable().max(10));
/// # let b = vars.add(variable());
/// let my_inequality = constraint!(a + b >= 3 * b - a);
/// ```
///
/// ## Full example
///
/// ```
/// # use float_eq::assert_float_eq;
/// use good_lp::*;
///
/// let mut vars = variables!();
/// let a = vars.add(variable().max(10));
/// let b = vars.add(variable());
/// let solution = vars
///     .maximise(a + b)
///     .using(default_solver)
///     .with(constraint!(a - 5 <= b / 2))
///     .with(constraint!(b == a))
///     .solve().unwrap();
/// assert_float_eq!(10., solution.value(a), abs<=1e-8);
/// assert_float_eq!(10., solution.value(b), abs<=1e-8);
/// ```
#[macro_export]
macro_rules! constraint {
    ([$($left:tt)*] <= $($right:tt)*) => {
        $crate::constraint::leq($($left)*, $($right)*)
    };
    ([$($left:tt)*] >= $($right:tt)*) => {
        $crate::constraint::geq($($left)*, $($right)*)
    };
    ([$($left:tt)*] == $($right:tt)*) => {
        $crate::constraint::eq($($left)*, $($right)*)
    };
    // Stop condition: all token have been processed
    ([$($left:tt)*]) => {
        $($left:tt)*
    };
    // The next token is not a special one
    ([$($left:tt)*] $next:tt $($right:tt)*) => {
        constraint!([$($left)* $next] $($right)*)
    };
    // Initial rule: start the recursive calls
    ($($all:tt)*) => {
        constraint!([] $($all)*)
    };
}

#[derive(Clone)]
/// A constraint reference contains the sequence id and direction of a constraint within the problem.
pub struct ConstraintReference {
    pub(crate) index: usize,
    // This metadata is consumed by the dual-capable solver integrations.
    #[allow(dead_code)]
    direction: ConstraintDirection,
}

impl ConstraintReference {
    pub(crate) fn with_direction(index: usize, direction: ConstraintDirection) -> Self {
        Self { index, direction }
    }

    #[cfg(any(feature = "highs", feature = "clarabel"))]
    pub(crate) fn dual_sign(&self) -> f64 {
        self.direction.dual_sign()
    }
}

impl PartialEq for ConstraintReference {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Debug for ConstraintReference {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintReference")
            .field("index", &self.index)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::variables;
    #[test]
    fn test_leq() {
        let mut vars = variables!();
        let v0 = vars.add_variable();
        let v1 = vars.add_variable();
        let f = format!("{:?}", (3. - v0) >> v1);
        assert!(["v0 + v1 <= 3", "v1 + v0 <= 3"].contains(&&*f), "{}", f)
    }
}
