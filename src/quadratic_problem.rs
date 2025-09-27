use crate::quadratic_expression::QuadraticAffineExpression;
use crate::solvers::ObjectiveDirection;
use crate::variable::ProblemVariables;

/// A problem with a quadratic objective function
pub struct QuadraticUnsolvedProblem {
    /// The quadratic objective function to optimize
    pub objective: QuadraticAffineExpression,
    /// Whether to maximize or minimize the objective
    pub direction: ObjectiveDirection,
    /// The variables in the problem
    pub variables: ProblemVariables,
}

/// Trait for quadratic expressions that can be used as objective functions
pub trait IntoQuadraticExpression {
    /// Transform the value into a concrete QuadraticAffineExpression
    fn into_quadratic_expression(self) -> QuadraticAffineExpression;
}

impl IntoQuadraticExpression for QuadraticAffineExpression {
    fn into_quadratic_expression(self) -> QuadraticAffineExpression {
        self
    }
}

impl IntoQuadraticExpression for crate::QuadraticExpression {
    fn into_quadratic_expression(self) -> QuadraticAffineExpression {
        QuadraticAffineExpression::from_quadratic(self)
    }
}

impl<T: crate::IntoAffineExpression> IntoQuadraticExpression for T {
    fn into_quadratic_expression(self) -> QuadraticAffineExpression {
        QuadraticAffineExpression::from_affine(self)
    }
}

impl ProblemVariables {
    /// Create a quadratic optimization problem to maximize the given quadratic objective
    #[cfg(feature = "enable_quadratic")]
    pub fn maximise_quadratic<E: IntoQuadraticExpression>(
        self,
        objective: E,
    ) -> QuadraticUnsolvedProblem {
        let objective_expr = objective.into_quadratic_expression();
        // Validate that objective variables are within bounds
        for (var, _) in objective_expr.linear.coefficients.iter() {
            assert!(
                var.index() < self.len(),
                "Variable in objective function is not part of this problem"
            );
        }
        for (pair, _) in objective_expr.quadratic.quadratic_coefficients.iter() {
            assert!(
                pair.var1.index() < self.len() && pair.var2.index() < self.len(),
                "Variable in quadratic objective function is not part of this problem"
            );
        }
        QuadraticUnsolvedProblem {
            objective: objective_expr,
            direction: ObjectiveDirection::Maximisation,
            variables: self,
        }
    }

    /// Create a quadratic optimization problem to minimize the given quadratic objective
    #[cfg(feature = "enable_quadratic")]
    pub fn minimise_quadratic<E: IntoQuadraticExpression>(
        self,
        objective: E,
    ) -> QuadraticUnsolvedProblem {
        let objective_expr = objective.into_quadratic_expression();
        // Validate that objective variables are within bounds
        for (var, _) in objective_expr.linear.coefficients.iter() {
            assert!(
                var.index() < self.len(),
                "Variable in objective function is not part of this problem"
            );
        }
        for (pair, _) in objective_expr.quadratic.quadratic_coefficients.iter() {
            assert!(
                pair.var1.index() < self.len() && pair.var2.index() < self.len(),
                "Variable in quadratic objective function is not part of this problem"
            );
        }
        QuadraticUnsolvedProblem {
            objective: objective_expr,
            direction: ObjectiveDirection::Minimisation,
            variables: self,
        }
    }
}

impl QuadraticUnsolvedProblem {
    /// Solve this quadratic problem using the given solver function
    #[cfg(all(feature = "clarabel", feature = "enable_quadratic"))]
    pub fn using<F, S>(self, solver_factory: F) -> S
    where
        F: FnOnce(Self) -> S,
    {
        solver_factory(self)
    }
}
