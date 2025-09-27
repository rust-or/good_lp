use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use fnv::FnvHashMap as HashMap;

use crate::affine_expression_trait::IntoAffineExpression;

use crate::expression::{Expression, LinearExpression};
use crate::variable::{FormatWithVars, Variable};
use crate::Solution;

/// Represents a quadratic term as a pair of variables and their coefficient
/// For example, 3*x*y would be represented as QuadraticTerm { var1: x, var2: y, coeff: 3.0 }
/// For x^2, both var1 and var2 would be x
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VariablePair {
    /// First variable in the pair
    pub var1: Variable,
    /// Second variable in the pair
    pub var2: Variable,
}

impl VariablePair {
    /// Create a new variable pair, ensuring consistent ordering for commutativity (x*y = y*x)
    pub fn new(var1: Variable, var2: Variable) -> Self {
        // Ensure consistent ordering for commutativity (x*y = y*x)
        if var1.index() <= var2.index() {
            VariablePair { var1, var2 }
        } else {
            VariablePair {
                var1: var2,
                var2: var1,
            }
        }
    }

    /// Create a variable pair representing a square term (var^2)
    pub fn square(var: Variable) -> Self {
        VariablePair {
            var1: var,
            var2: var,
        }
    }
}

/// A quadratic expression without linear or constant components
pub struct QuadraticExpression {
    pub(crate) quadratic_coefficients: HashMap<VariablePair, f64>,
}

impl QuadraticExpression {
    /// Create a new empty quadratic expression
    pub fn new() -> Self {
        QuadraticExpression {
            quadratic_coefficients: HashMap::default(),
        }
    }

    /// Create a quadratic expression with preallocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        QuadraticExpression {
            quadratic_coefficients: HashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Add a quadratic term to this expression
    pub fn add_quadratic_term(&mut self, var1: Variable, var2: Variable, coefficient: f64) {
        let pair = VariablePair::new(var1, var2);
        *self.quadratic_coefficients.entry(pair).or_default() += coefficient;
    }

    /// Get the coefficient for a specific variable pair
    pub fn get_quadratic_coefficient(&self, var1: Variable, var2: Variable) -> f64 {
        let pair = VariablePair::new(var1, var2);
        self.quadratic_coefficients
            .get(&pair)
            .copied()
            .unwrap_or(0.0)
    }

    /// Check if the expression is empty (all coefficients are zero)
    pub fn is_empty(&self) -> bool {
        self.quadratic_coefficients.is_empty()
            || self
                .quadratic_coefficients
                .values()
                .all(|&c| c.abs() < f64::EPSILON)
    }

    /// Iterate over all non-zero quadratic terms
    pub fn iter(&self) -> impl Iterator<Item = (VariablePair, f64)> + '_ {
        self.quadratic_coefficients
            .iter()
            .filter_map(|(&pair, &coeff)| {
                if coeff.abs() >= f64::EPSILON {
                    Some((pair, coeff))
                } else {
                    None
                }
            })
    }

    /// Evaluate the quadratic expression given variable values
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        self.quadratic_coefficients
            .iter()
            .map(|(pair, &coeff)| {
                let val1 = values.value(pair.var1);
                let val2 = values.value(pair.var2);
                coeff * val1 * val2
            })
            .sum()
    }
}

impl Default for QuadraticExpression {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for QuadraticExpression {
    fn clone(&self) -> Self {
        QuadraticExpression {
            quadratic_coefficients: self.quadratic_coefficients.clone(),
        }
    }
}

/// A complete quadratic expression containing quadratic, linear, and constant terms
/// Represents expressions like: x^2 + 2*x*y + 3*x + 4
pub struct QuadraticAffineExpression {
    pub(crate) quadratic: QuadraticExpression,
    pub(crate) linear: LinearExpression,
    pub(crate) constant: f64,
}

impl QuadraticAffineExpression {
    /// Create a new empty quadratic affine expression
    pub fn new() -> Self {
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::new(),
            linear: LinearExpression {
                coefficients: HashMap::default(),
            },
            constant: 0.0,
        }
    }

    /// Create a quadratic affine expression with preallocated capacity
    pub fn with_capacity(linear_capacity: usize, quadratic_capacity: usize) -> Self {
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::with_capacity(quadratic_capacity),
            linear: LinearExpression {
                coefficients: HashMap::with_capacity_and_hasher(
                    linear_capacity,
                    Default::default(),
                ),
            },
            constant: 0.0,
        }
    }

    /// Create from an affine expression
    pub fn from_affine<E: IntoAffineExpression>(expr: E) -> Self {
        let constant = expr.constant();
        let linear_coeffs = expr.linear_coefficients().into_iter().collect();
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::new(),
            linear: LinearExpression {
                coefficients: linear_coeffs,
            },
            constant,
        }
    }

    /// Create from a quadratic expression
    pub fn from_quadratic(expr: QuadraticExpression) -> Self {
        QuadraticAffineExpression {
            quadratic: expr,
            linear: LinearExpression {
                coefficients: HashMap::default(),
            },
            constant: 0.0,
        }
    }

    /// Add a quadratic term
    pub fn add_quadratic_term(&mut self, var1: Variable, var2: Variable, coefficient: f64) {
        self.quadratic.add_quadratic_term(var1, var2, coefficient);
    }

    /// Add a linear term
    pub fn add_linear_term(&mut self, var: Variable, coefficient: f64) {
        *self.linear.coefficients.entry(var).or_default() += coefficient;
    }

    /// Add a constant term
    pub fn add_constant(&mut self, value: f64) {
        self.constant += value;
    }

    /// Get the linear part as an Expression
    pub fn linear_part(&self) -> Expression {
        Expression {
            linear: LinearExpression {
                coefficients: self.linear.coefficients.clone(),
            },
            constant: self.constant,
        }
    }

    /// Check if the expression contains any quadratic terms
    pub fn is_purely_affine(&self) -> bool {
        self.quadratic.is_empty()
    }

    /// Evaluate the complete expression
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        self.quadratic.eval_with(values) + self.linear_part().eval_with(values)
    }

    /// Get the coefficient of a quadratic term
    pub fn get_quadratic_coefficient(&self, var1: Variable, var2: Variable) -> f64 {
        self.quadratic.get_quadratic_coefficient(var1, var2)
    }

    /// Get the coefficient of a linear term
    pub fn get_linear_coefficient(&self, var: Variable) -> f64 {
        self.linear.coefficients.get(&var).copied().unwrap_or(0.0)
    }

    /// Get the constant term
    pub fn get_constant(&self) -> f64 {
        self.constant
    }

    /// Get a reference to the quadratic part
    pub fn quadratic_part(&self) -> &QuadraticExpression {
        &self.quadratic
    }

    /// Creates a constraint indicating that this quadratic expression
    /// is lesser than or equal to the right hand side
    pub fn leq<RHS>(self, rhs: RHS) -> QuadraticConstraint
    where
        Self: Sub<RHS, Output = Self>,
    {
        let diff = self - rhs;
        QuadraticConstraint {
            expression: diff,
            is_equality: false,
        }
    }

    /// Creates a constraint indicating that this quadratic expression
    /// is greater than or equal to the right hand side
    pub fn geq<RHS>(self, rhs: RHS) -> QuadraticConstraint
    where
        RHS: Sub<Self, Output = Self>,
    {
        let diff = rhs - self;
        QuadraticConstraint {
            expression: diff,
            is_equality: false,
        }
    }

    /// Creates a constraint indicating that this quadratic expression
    /// is equal to the right hand side
    pub fn eq<RHS>(self, rhs: RHS) -> QuadraticConstraint
    where
        Self: Sub<RHS, Output = Self>,
    {
        let diff = self - rhs;
        QuadraticConstraint {
            expression: diff,
            is_equality: true,
        }
    }
}

impl Default for QuadraticAffineExpression {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for QuadraticAffineExpression {
    fn clone(&self) -> Self {
        QuadraticAffineExpression {
            quadratic: self.quadratic.clone(),
            linear: self.linear.clone(),
            constant: self.constant,
        }
    }
}

/// A quadratic constraint
pub struct QuadraticConstraint {
    /// The quadratic expression that defines the constraint
    pub expression: QuadraticAffineExpression,
    /// Whether this is an equality constraint (true) or inequality constraint (false)
    pub is_equality: bool,
}

// Conversion implementations
impl From<f64> for QuadraticAffineExpression {
    fn from(value: f64) -> Self {
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::new(),
            linear: LinearExpression {
                coefficients: HashMap::default(),
            },
            constant: value,
        }
    }
}

impl From<Variable> for QuadraticAffineExpression {
    fn from(var: Variable) -> Self {
        let mut linear_coeffs = HashMap::default();
        linear_coeffs.insert(var, 1.0);
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::new(),
            linear: LinearExpression {
                coefficients: linear_coeffs,
            },
            constant: 0.0,
        }
    }
}

impl From<Expression> for QuadraticAffineExpression {
    fn from(expr: Expression) -> Self {
        QuadraticAffineExpression {
            quadratic: QuadraticExpression::new(),
            linear: expr.linear,
            constant: expr.constant,
        }
    }
}

impl From<QuadraticExpression> for QuadraticAffineExpression {
    fn from(quad_expr: QuadraticExpression) -> Self {
        QuadraticAffineExpression::from_quadratic(quad_expr)
    }
}

// Arithmetic operations for QuadraticExpression
impl<N: Into<f64>> MulAssign<N> for QuadraticExpression {
    fn mul_assign(&mut self, rhs: N) {
        let factor = rhs.into();
        for value in self.quadratic_coefficients.values_mut() {
            *value *= factor;
        }
    }
}

impl<N: Into<f64>> Mul<N> for QuadraticExpression {
    type Output = QuadraticExpression;

    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<N: Into<f64>> Div<N> for QuadraticExpression {
    type Output = QuadraticExpression;

    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

impl AddAssign<QuadraticExpression> for QuadraticExpression {
    fn add_assign(&mut self, rhs: QuadraticExpression) {
        for (pair, coeff) in rhs.quadratic_coefficients {
            *self.quadratic_coefficients.entry(pair).or_default() += coeff;
        }
    }
}

impl Add<QuadraticExpression> for QuadraticExpression {
    type Output = QuadraticExpression;

    fn add(mut self, rhs: QuadraticExpression) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl SubAssign<QuadraticExpression> for QuadraticExpression {
    fn sub_assign(&mut self, rhs: QuadraticExpression) {
        for (pair, coeff) in rhs.quadratic_coefficients {
            *self.quadratic_coefficients.entry(pair).or_default() -= coeff;
        }
    }
}

impl Sub<QuadraticExpression> for QuadraticExpression {
    type Output = QuadraticExpression;

    fn sub(mut self, rhs: QuadraticExpression) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl Neg for QuadraticExpression {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

// Arithmetic operations for QuadraticAffineExpression
impl<N: Into<f64>> MulAssign<N> for QuadraticAffineExpression {
    fn mul_assign(&mut self, rhs: N) {
        let factor = rhs.into();
        self.quadratic.mul_assign(factor);
        for value in self.linear.coefficients.values_mut() {
            *value *= factor;
        }
        self.constant *= factor;
    }
}

impl<N: Into<f64>> Mul<N> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<N: Into<f64>> Div<N> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

impl AddAssign<QuadraticAffineExpression> for QuadraticAffineExpression {
    fn add_assign(&mut self, rhs: QuadraticAffineExpression) {
        self.quadratic.add_assign(rhs.quadratic);
        for (var, coeff) in rhs.linear.coefficients {
            *self.linear.coefficients.entry(var).or_default() += coeff;
        }
        self.constant += rhs.constant;
    }
}

impl Add<QuadraticAffineExpression> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn add(mut self, rhs: QuadraticAffineExpression) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl SubAssign<QuadraticAffineExpression> for QuadraticAffineExpression {
    fn sub_assign(&mut self, rhs: QuadraticAffineExpression) {
        self.quadratic.sub_assign(rhs.quadratic);
        for (var, coeff) in rhs.linear.coefficients {
            *self.linear.coefficients.entry(var).or_default() -= coeff;
        }
        self.constant -= rhs.constant;
    }
}

impl Sub<QuadraticAffineExpression> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn sub(mut self, rhs: QuadraticAffineExpression) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl Neg for QuadraticAffineExpression {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

// Mixed type operations
impl<E: IntoAffineExpression> AddAssign<E> for QuadraticAffineExpression {
    fn add_assign(&mut self, rhs: E) {
        let constant = rhs.constant();
        for (var, coeff) in rhs.linear_coefficients() {
            *self.linear.coefficients.entry(var).or_default() += coeff;
        }
        self.constant += constant;
    }
}

impl<E: IntoAffineExpression> Add<E> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn add(mut self, rhs: E) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl<E: IntoAffineExpression> SubAssign<E> for QuadraticAffineExpression {
    fn sub_assign(&mut self, rhs: E) {
        let constant = rhs.constant();
        for (var, coeff) in rhs.linear_coefficients() {
            *self.linear.coefficients.entry(var).or_default() -= coeff;
        }
        self.constant -= constant;
    }
}

impl<E: IntoAffineExpression> Sub<E> for QuadraticAffineExpression {
    type Output = QuadraticAffineExpression;

    fn sub(mut self, rhs: E) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

// Multiplication operations that create quadratic terms
impl Mul<Variable> for Variable {
    type Output = QuadraticExpression;

    fn mul(self, rhs: Variable) -> Self::Output {
        let mut quadratic = QuadraticExpression::new();
        quadratic.add_quadratic_term(self, rhs, 1.0);
        quadratic
    }
}

// Specific implementations for Variable multiplication with other types
impl Mul<Expression> for Variable {
    type Output = QuadraticAffineExpression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let mut result = QuadraticAffineExpression::new();
        let constant = rhs.constant;

        // Linear term (constant * variable)
        if constant.abs() >= f64::EPSILON {
            result.add_linear_term(self, constant);
        }

        // Quadratic terms (variable * other_variables)
        for (var, coeff) in rhs.linear.coefficients {
            result.add_quadratic_term(self, var, coeff);
        }

        result
    }
}

impl Mul<Variable> for Expression {
    type Output = QuadraticAffineExpression;

    fn mul(self, rhs: Variable) -> Self::Output {
        rhs * self
    }
}

// Expression multiplication (returns quadratic expression)
impl Mul<Expression> for Expression {
    type Output = QuadraticAffineExpression;

    fn mul(self, rhs: Expression) -> Self::Output {
        let mut result = QuadraticAffineExpression::new();

        let c1 = self.constant;
        let c2 = rhs.constant;

        // Constant term (c1 * c2)
        result.constant = c1 * c2;

        // Linear terms from self * constant of rhs
        for (var, coeff) in self.linear.coefficients.iter() {
            result.add_linear_term(*var, *coeff * c2);
        }

        // Linear terms from constant of self * rhs
        for (var, coeff) in rhs.linear.coefficients.iter() {
            result.add_linear_term(*var, c1 * *coeff);
        }

        // Quadratic terms (variable from self * variable from rhs)
        for (var1, coeff1) in self.linear.coefficients.iter() {
            for (var2, coeff2) in rhs.linear.coefficients.iter() {
                result.add_quadratic_term(*var1, *var2, *coeff1 * *coeff2);
            }
        }

        result
    }
}

// Numeric multiplication
macro_rules! impl_mul_numeric {
    ($($t:ty),*) => {$(
        impl Mul<QuadraticExpression> for $t {
            type Output = QuadraticExpression;

            fn mul(self, mut rhs: QuadraticExpression) -> Self::Output {
                rhs *= self;
                rhs
            }
        }

        impl Mul<QuadraticAffineExpression> for $t {
            type Output = QuadraticAffineExpression;

            fn mul(self, mut rhs: QuadraticAffineExpression) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    )*}
}
impl_mul_numeric!(f64, i32, f32, u32, u16, u8, i16, i8);

// Display and formatting support
impl FormatWithVars for QuadraticExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let mut first = true;
        for (pair, coeff) in self.iter() {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }

            if (coeff - 1.).abs() > f64::EPSILON {
                write!(f, "{} ", coeff)?;
            }

            if pair.var1 == pair.var2 {
                // Square term: x^2
                variable_format(f, pair.var1)?;
                write!(f, "²")?;
            } else {
                // Cross term: x*y
                variable_format(f, pair.var1)?;
                write!(f, "*")?;
                variable_format(f, pair.var2)?;
            }
        }

        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl Debug for QuadraticExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

impl FormatWithVars for QuadraticAffineExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let has_quadratic = !self.quadratic.is_empty();
        let has_linear = !self.linear.coefficients.is_empty();
        let has_constant = self.constant.abs() >= f64::EPSILON;

        if !has_quadratic && !has_linear && !has_constant {
            return write!(f, "0");
        }

        let mut first = true;

        // Write quadratic terms
        if has_quadratic {
            self.quadratic.format_with(f, &mut variable_format)?;
            first = false;
        }

        // Write linear terms
        for (&var, &coeff) in &self.linear.coefficients {
            if coeff.abs() >= f64::EPSILON {
                if !first {
                    write!(f, " + ")?;
                }
                first = false;

                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, var)?;
            }
        }

        // Write constant term
        if has_constant {
            if !first {
                write!(f, " + ")?;
            }
            write!(f, "{}", self.constant)?;
        }

        Ok(())
    }
}

impl Debug for QuadraticAffineExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}
