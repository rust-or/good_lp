use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Constraint, Solution};
use crate::constraint;
use crate::variable::{Variable, FormatWithVars};
use std::fmt::{Debug, Formatter};

pub(crate) struct LinearExpression<F> {
    pub(crate) coefficients: HashMap<Variable<F>, f64>
}

impl<F> FormatWithVars<F> for LinearExpression<F> {
    fn format_with<FUN>(
        &self,
        f: &mut Formatter<'_>,
        variable_format: FUN,
    ) -> std::fmt::Result
        where FUN: Fn(&mut Formatter<'_>, Variable<F>) -> std::fmt::Result {
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

pub struct Expression<F> {
    pub(crate) linear: LinearExpression<F>,
    pub(crate) constant: f64,
}

impl<F> PartialEq for Expression<F> {
    fn eq(&self, other: &Self) -> bool {
        self.constant.eq(&other.constant) &&
            self.linear.coefficients.eq(&other.linear.coefficients)
    }
}

impl<F> Clone for Expression<F> {
    fn clone(&self) -> Self {
        Expression {
            linear: LinearExpression { coefficients: self.linear.coefficients.clone() },
            constant: self.constant,
        }
    }
}

impl<T> Debug for Expression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

impl<F> Default for Expression<F> {
    fn default() -> Self {
        let coefficients = HashMap::with_capacity(0);
        Expression { linear: LinearExpression { coefficients }, constant: 0. }
    }
}

impl<F> Expression<F> {
    /// Creates a constraint indicating that this expression
    /// is lesser than or equal to the right hand side
    pub fn leq<RHS: Into<Expression<F>>>(self, rhs: RHS) -> Constraint<F> {
        constraint::leq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is greater than or equal to the right hand side
    pub fn geq<RHS: Into<Expression<F>>>(self, rhs: RHS) -> Constraint<F> {
        constraint::geq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is greater than or equal to the right hand side
    pub fn eq<RHS: Into<Expression<F>>>(self, rhs: RHS) -> Constraint<F> {
        constraint::eq(self, rhs)
    }

    // Computes self + (a * b)
    #[inline]
    pub fn add_mul<N: Into<f64>>(&mut self, a: N, b: &Expression<F>) {
        let factor = a.into();
        for (var, value) in &b.linear.coefficients {
            *self.linear.coefficients.entry(*var).or_default() += factor * value
        }
        self.constant += factor * b.constant;
    }

    /// Evaluate the concrete value of the expression, given the values of the variables
    ///
    /// ## Example
    ///
    /// ```rust
    /// use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};
    /// let mut vars = variables!();
    /// let a = vars.add(variable().max(1));
    /// let b = vars.add(variable().max(4));
    /// let objective = a + b;
    /// let solution = vars.maximise(objective.clone()).using(coin_cbc).solve()?;
    /// assert_eq!(objective.eval_with(&solution), 5.);
    /// # use good_lp::ResolutionError;
    /// # Ok::<_, ResolutionError>(())
    /// ```
    pub fn eval_with<S: Solution<F>>(&self, values: &S) -> f64 {
        self.constant +
            self.linear.coefficients
                .iter()
                .map(|(&var, coefficient)|
                    coefficient * values.value(var)
                ).sum::<f64>()
    }
}

pub fn add_mul<F, LHS: Into<Expression<F>>, RHS: Into<Expression<F>>>(lhs: LHS, rhs: RHS, factor: f64) -> Expression<F> {
    let mut result = lhs.into();
    result.add_mul(factor, &rhs.into());
    result
}

pub fn sub<F, LHS: Into<Expression<F>>, RHS: Into<Expression<F>>>(lhs: LHS, rhs: RHS) -> Expression<F> {
    add_mul(lhs, rhs, -1.)
}


pub fn add<F, LHS: Into<Expression<F>>, RHS: Into<Expression<F>>>(lhs: LHS, rhs: RHS) -> Expression<F> {
    add_mul(lhs, rhs, 1.)
}

impl<F> FormatWithVars<F> for Expression<F> {
    fn format_with<FUN>(
        &self,
        f: &mut Formatter<'_>, variable_format: FUN,
    ) -> std::fmt::Result
        where FUN: Fn(&mut Formatter<'_>, Variable<F>) -> std::fmt::Result {
        self.linear.format_with(f, variable_format)?;
        write!(f, " + {}", self.constant)
    }
}

impl<'a, F> AddAssign<&'a Expression<F>> for Expression<F> {
    fn add_assign(&mut self, rhs: &'a Expression<F>) {
        self.add_mul(1., rhs)
    }
}

impl<'a, F> Add<&'a Expression<F>> for Expression<F> {
    type Output = Expression<F>;

    fn add(mut self, rhs: &'a Expression<F>) -> Self::Output {
        self += rhs;
        self
    }
}

impl<'a, F> SubAssign<&'a Expression<F>> for Expression<F> {
    fn sub_assign(&mut self, rhs: &'a Expression<F>) {
        self.add_mul(-1., rhs)
    }
}

impl<'a, F> Sub<&'a Expression<F>> for Expression<F> {
    type Output = Expression<F>;

    fn sub(mut self, rhs: &'a Expression<F>) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<F, RHS: Into<Expression<F>>> AddAssign<RHS> for Expression<F> {
    fn add_assign(&mut self, rhs: RHS) {
        self.add_assign(&rhs.into());
    }
}

impl<F, RHS: Into<Expression<F>>> SubAssign<RHS> for Expression<F> {
    fn sub_assign(&mut self, rhs: RHS) {
        self.sub_assign(&rhs.into());
    }
}

impl<T> Neg for Expression<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

impl<F, N: Into<f64>> MulAssign<N> for Expression<F> {
    fn mul_assign(&mut self, rhs: N) {
        let factor = rhs.into();
        for value in self.linear.coefficients.values_mut() {
            *value *= factor
        }
        self.constant *= factor
    }
}

impl<F, N: Into<f64>> Mul<N> for Expression<F> {
    type Output = Expression<F>;

    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<F, N: Into<f64>> Div<N> for Expression<F> {
    type Output = Expression<F>;

    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

impl<F> From<Variable<F>> for Expression<F> {
    fn from(var: Variable<F>) -> Self {
        Expression::from(&var)
    }
}

impl<'a, F> From<&'a Variable<F>> for Expression<F> {
    fn from(var: &'a Variable<F>) -> Self {
        let mut coefficients = HashMap::with_capacity(1);
        coefficients.insert(*var, 1.);
        Expression { linear: LinearExpression { coefficients }, constant: 0.0 }
    }
}

impl<F, N: Into<f64>> From<N> for Expression<F> {
    fn from(constant: N) -> Self {
        let coefficients = HashMap::with_capacity(0);
        let constant = constant.into();
        Expression { linear: LinearExpression { coefficients }, constant }
    }
}

impl<F> Mul<Expression<F>> for f64 {
    type Output = Expression<F>;

    fn mul(self, mut rhs: Expression<F>) -> Self::Output {
        rhs *= self;
        rhs
    }
}

impl<F> Mul<Expression<F>> for i32 {
    type Output = Expression<F>;

    fn mul(self, mut rhs: Expression<F>) -> Self::Output {
        rhs *= self;
        rhs
    }
}

macro_rules! impl_ops_local {
    ($($typename:ident $(< $param:ident >)? : $other:ident $(< $other_param:ident >)?),*) => {$(
        impl<F $(, $other: Into<Expression<$param>>)?>
            Sub<$other $(< $other_param >)?>
        for $typename$(< $param >)? {
            type Output = Expression<F>;
            fn sub(self, rhs: $other $(< $other_param >)?) -> Self::Output { sub(self, rhs) }
        }

        impl<F $(, $other: Into<Expression<$param>>)?>
            Add<$other $(< $other_param >)?>
        for $typename$(< $param >)? {
            type Output = Expression<F>;
            fn add(self, rhs: $other $(< $other_param >)?) -> Self::Output { add(self, rhs) }
        }
    )*}
}

impl_ops_local!(
    Expression<F> : RHS,
    Variable<F> : RHS,
    f64: Expression<F>,
    f64: Variable<F>,
    i32: Expression<F>,
    i32: Variable<F>
);

impl<'a, F, A> std::iter::Sum<A> for Expression<F>
    where Expression<F>: From<A> {
    fn sum<I: Iterator<Item=A>>(iter: I) -> Self {
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