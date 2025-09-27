use std::fmt::Debug;
use std::ops::Mul;

use fnv::FnvHashMap as HashMap;

use crate::affine_expression_trait::IntoAffineExpression;
use crate::expression::{Expression, LinearExpression};
use crate::variable::Variable;
use crate::Solution;

/// Represents a pair of variables in a quadratic term
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
        if var1.index() <= var2.index() {
            VariablePair { var1, var2 }
        } else {
            VariablePair {
                var1: var2,
                var2: var1,
            }
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

    /// Add a quadratic term to this expression
    pub fn add_quadratic_term(&mut self, var1: Variable, var2: Variable, coefficient: f64) {
        let pair = VariablePair::new(var1, var2);
        *self.quadratic_coefficients.entry(pair).or_default() += coefficient;
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

    /// Evaluate the complete expression
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        self.quadratic.eval_with(values) 
            + self.linear.coefficients.iter().map(|(&var, &coeff)| coeff * values.value(var)).sum::<f64>()
            + self.constant
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

impl Debug for QuadraticAffineExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuadraticAffineExpression {{ quadratic: {:?}, linear: {:?}, constant: {} }}", 
               self.quadratic, self.linear, self.constant)
    }
}

impl Debug for QuadraticExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuadraticExpression")
            .field("quadratic_coefficients", &self.quadratic_coefficients)
            .finish()
    }
}

/// A quadratic constraint
pub struct QuadraticConstraint {
    /// The quadratic expression that defines the constraint
    pub expression: QuadraticAffineExpression,
    /// Whether this is an equality constraint (true) or inequality constraint (false)
    pub is_equality: bool,
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

// Essential multiplication operations for creating quadratic terms
impl Mul<Variable> for Variable {
    type Output = QuadraticExpression;

    fn mul(self, rhs: Variable) -> Self::Output {
        let mut quadratic = QuadraticExpression::new();
        quadratic.add_quadratic_term(self, rhs, 1.0);
        quadratic
    }
}
