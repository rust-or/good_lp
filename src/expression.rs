use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::constraint;
use crate::variable::{FormatWithVars, Variable};
use crate::{Constraint, Solution};
use std::fmt::{Debug, Formatter};

pub(crate) struct LinearExpression {
    pub(crate) coefficients: HashMap<Variable, f64>,
}

impl FormatWithVars for LinearExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: Fn(&mut Formatter<'_>, Variable) -> std::fmt::Result,
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
        let coefficients = HashMap::with_capacity(0);
        Expression {
            linear: LinearExpression { coefficients },
            constant: 0.,
        }
    }
}

impl Expression {
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
    /// is greater than or equal to the right hand side
    pub fn eq<RHS>(self, rhs: RHS) -> Constraint
    where
        Expression: Sub<RHS, Output = Expression>,
    {
        constraint::eq(self, rhs)
    }

    /// Performs self = self + (a * b)
    #[inline]
    pub fn add_mul<N: Into<f64>>(&mut self, a: N, b: &Expression) {
        let factor = a.into();
        for (var, value) in &b.linear.coefficients {
            *self.linear.coefficients.entry(*var).or_default() += factor * value
        }
        self.constant += factor * b.constant;
    }

    /// Evaluate the concrete value of the expression, given the values of the variables
    ///
    /// ## Examples
    /// ### Evaluate an expression using a solution
    ///
    /// ```rust
    /// # #[cfg(feature = "coin_cbc")] {
    /// use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
    /// let mut vars = variables!();
    /// let a = vars.add(variable().max(1));
    /// let b = vars.add(variable().max(4));
    /// let objective = a + b;
    /// let solution = vars.maximise(objective.clone()).using(coin_cbc).solve()?;
    /// assert_eq!(objective.eval_with(&solution), 5.);
    /// # }
    /// # use good_lp::ResolutionError;
    /// # Ok::<_, ResolutionError>(())
    /// ```
    ///
    /// ### Evaluate an expression with a HashMap
    /// A [HashMap] is a valid [Solution]
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
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        self.constant
            + self
                .linear
                .coefficients
                .iter()
                .map(|(&var, coefficient)| coefficient * values.value(var))
                .sum::<f64>()
    }
}

pub fn add_mul<LHS: Into<Expression>, RHS: Into<Expression>>(
    lhs: LHS,
    rhs: RHS,
    factor: f64,
) -> Expression {
    let mut result = lhs.into();
    result.add_mul(factor, &rhs.into());
    result
}

pub fn sub<LHS: Into<Expression>, RHS: Into<Expression>>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, -1.)
}

pub fn add<LHS: Into<Expression>, RHS: Into<Expression>>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, 1.)
}

impl FormatWithVars for Expression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: Fn(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        self.linear.format_with(f, variable_format)?;
        write!(f, " + {}", self.constant)
    }
}

impl<'a> AddAssign<&'a Expression> for Expression {
    fn add_assign(&mut self, rhs: &'a Expression) {
        self.add_mul(1., rhs)
    }
}

impl<'a> Add<&'a Expression> for Expression {
    type Output = Expression;

    fn add(mut self, rhs: &'a Expression) -> Self::Output {
        self += rhs;
        self
    }
}

impl<'a> SubAssign<&'a Expression> for Expression {
    fn sub_assign(&mut self, rhs: &'a Expression) {
        self.add_mul(-1., rhs)
    }
}

impl<'a> Sub<&'a Expression> for Expression {
    type Output = Expression;

    fn sub(mut self, rhs: &'a Expression) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<RHS: Into<Expression>> AddAssign<RHS> for Expression {
    fn add_assign(&mut self, rhs: RHS) {
        self.add_assign(&rhs.into());
    }
}

impl<RHS: Into<Expression>> SubAssign<RHS> for Expression {
    fn sub_assign(&mut self, rhs: RHS) {
        self.sub_assign(&rhs.into());
    }
}

impl Neg for Expression {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

impl<N: Into<f64>> MulAssign<N> for Expression {
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

    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<N: Into<f64>> Div<N> for Expression {
    type Output = Expression;

    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

impl From<Variable> for Expression {
    fn from(var: Variable) -> Self {
        Expression::from(&var)
    }
}

impl<'a> From<&'a Variable> for Expression {
    fn from(var: &'a Variable) -> Self {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(*var, 1.);
        Expression {
            linear: LinearExpression { coefficients },
            constant: 0.0,
        }
    }
}

impl<N: Into<f64>> From<N> for Expression {
    fn from(constant: N) -> Self {
        let coefficients = HashMap::with_capacity(0);
        let constant = constant.into();
        Expression {
            linear: LinearExpression { coefficients },
            constant,
        }
    }
}

impl Mul<Expression> for f64 {
    type Output = Expression;

    fn mul(self, mut rhs: Expression) -> Self::Output {
        rhs *= self;
        rhs
    }
}

impl Mul<Expression> for i32 {
    type Output = Expression;

    fn mul(self, mut rhs: Expression) -> Self::Output {
        rhs *= self;
        rhs
    }
}

macro_rules! impl_ops_local {
    ($( $typename:ident : $([generic $generic:ident])? $other:ident),*) => {$(
        impl<$($generic: Into<Expression>)?> Sub<$other>
        for $typename {
            type Output = Expression;
            fn sub(self, rhs: $other) -> Self::Output { sub(self, rhs) }
        }

        impl<$($generic: Into<Expression>)?> Add<$other>
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

impl<'a, A> std::iter::Sum<A> for Expression
where
    Expression: From<A>,
{
    fn sum<I: Iterator<Item = A>>(iter: I) -> Self {
        let mut res = Expression::default();
        for i in iter {
            let expr = Expression::from(i);
            res.add_assign(expr)
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::variables;
    use std::collections::HashMap;

    #[test]
    fn expression_manipulation() {
        let mut vars = variables!();
        let v0 = vars.add_variable();
        let v1 = vars.add_variable();
        assert_eq!((3. - v0) - v1, (-1.) * v0 + (-1.) * v1 + 3.)
    }

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
