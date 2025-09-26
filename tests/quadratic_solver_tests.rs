//! Integration tests for quadratic programming functionality
//! These tests require both 'clarabel' and 'enable_quadratic' features

#[cfg(all(feature = "clarabel", feature = "enable_quadratic"))]
mod quadratic_integration_tests {
    use good_lp::{
        clarabel_quadratic, variables, QuadraticAffineExpression, ResolutionError, Solution,
        SolverModel,
    };

    #[test]
    fn test_constrained_quadratic_with_bounds() {
        variables! {
            vars:
                -2.0 <= x <= 2.0;
        }

        // Minimize x^2 - x
        let mut objective = QuadraticAffineExpression::new();

        objective.add_quadratic_term(x, x, 1.0);
        objective.add_linear_term(x, -1.0);

        // print objective for debugging
        println!("Objective: {:?}", objective);

        let solution = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .solve()
            .expect("Bounded quadratic should solve");

        println!("Solution: x={:.3}", solution.value(x));

        // Unconstrained minimum is at x=0.5, which satisfies the bounds
        assert!((solution.value(x) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_quadratic_with_equality_constraints() {
        variables! {
            vars:
                x;
                y;
                z;
        }

        // Minimize x^2 + y^2 + z^2 subject to x + y + z = 6 and x - y + 2z = 4
        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, 1.0);
        objective.add_quadratic_term(y, y, 1.0);
        objective.add_quadratic_term(z, z, 1.0);

        let solution = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .with((x + y + z).eq(6.0)) // First equality constraint
            .with((x - y + 2.0 * z).eq(4.0)) // Second equality constraint
            .solve()
            .expect("Quadratic with equality constraints should solve");

        // Global minimum is at x=2, y=2, z=2
        assert!((solution.value(x) - 2.0).abs() < 1e-3);
        assert!((solution.value(y) - 2.0).abs() < 1e-3);
        assert!((solution.value(z) - 2.0).abs() < 1e-3);

        // Verify constraints are satisfied
        let sum = solution.value(x) + solution.value(y) + solution.value(z);
        let diff = solution.value(x) - solution.value(y) + 2.0 * solution.value(z);

        assert!(
            (sum - 6.0).abs() < 1e-3,
            "First constraint violated: {} ≠ 6",
            sum
        );
        assert!(
            (diff - 4.0).abs() < 1e-3,
            "Second constraint violated: {} ≠ 4",
            diff
        );

        println!(
            "Solution: x={:.3}, y={:.3}, z={:.3}",
            solution.value(x),
            solution.value(y),
            solution.value(z)
        );
    }

    #[test]
    fn test_quadratic_with_inequality_constraints() {
        variables! {
            vars:
                x;
                y;
        }

        // Minimize x^2 + y^2 subject to x + y ≥ 4
        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, 1.0);
        objective.add_quadratic_term(y, y, 1.0);

        let solution = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .with((x + y).geq(4.0)) // inequality constraint
            .solve()
            .expect("Quadratic with inequality should solve");

        // By symmetry, the minimum occurs at x = 2, y = 2
        assert!((solution.value(x) - 2.0).abs() < 1e-3);
        assert!((solution.value(y) - 2.0).abs() < 1e-3);

        let sum = solution.value(x) + solution.value(y);
        assert!(sum >= 4.0 - 1e-3, "Constraint violated: {} < 4", sum);

        println!(
            "Solution: x={:.3}, y={:.3}",
            solution.value(x),
            solution.value(y)
        );
    }

    #[test]
    fn test_quadratic_with_mixed_constraints() {
        variables! {
            vars:
                x;
                y;
        }

        // Minimize (x-1)^2 + (y-3)^2 subject to x + y = 5, x ≥ 0
        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, 1.0);
        objective.add_quadratic_term(y, y, 1.0);
        objective.add_linear_term(x, -2.0);
        objective.add_linear_term(y, -6.0);
        objective.add_constant(10.0);

        let solution = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .with((x + y).eq(5.0)) // equality
            .with((1.0 *x).geq(0.0)) // inequality
            .solve()
            .expect("Quadratic with mixed constraints should solve");

        assert!((solution.value(x) - 1.5).abs() < 1e-3);
        assert!((solution.value(y) - 3.5).abs() < 1e-3);

        println!(
            "Solution: x={:.3}, y={:.3}",
            solution.value(x),
            solution.value(y)
        );
    }


    #[test]
    fn test_quadratic_maximization_problem() {
        variables! {
            vars:
                0.0 <= x <= 10.0;
                0.0 <= y <= 10.0;
        }

        // Maximize a concave quadratic: -x^2 - y^2 + 8x + 6y
        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, -1.0); // -x^2 (concave)
        objective.add_quadratic_term(y, y, -1.0); // -y^2 (concave)
        objective.add_linear_term(x, 8.0); // +8x
        objective.add_linear_term(y, 6.0); // +6y

        let solution = vars
            .maximise_quadratic(objective)
            .using(clarabel_quadratic)
            .solve()
            .expect("Quadratic maximization should solve");

        // Unconstrained maximum would be at x=4, y=3 (where derivatives are zero)
        assert!((solution.value(x) - 4.0).abs() < 1e-3);
        assert!((solution.value(y) - 3.0).abs() < 1e-3);

        // Verify bounds
        assert!(solution.value(x) >= -1e-6);
        assert!(solution.value(x) <= 10.0 + 1e-6);
        assert!(solution.value(y) >= -1e-6);
        assert!(solution.value(y) <= 10.0 + 1e-6);
    }

    #[test]
    fn test_infeasible_quadratic_problem() {
        variables! {
            vars:
                x;
                y;
        }

        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, 1.0);
        objective.add_quadratic_term(y, y, 1.0);

        // Create conflicting constraints: x + y >= 5 and x + y <= 2
        let result = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .with(x + y >> 5.0) // x + y >= 5
            .with(x + y << 2.0) // x + y <= 2 (conflicts with above)
            .solve();

        // Should return an infeasibility error
        match result {
            Err(ResolutionError::Infeasible) => {
                println!("Correctly detected infeasible problem");
            }
            _ => panic!("Should have detected infeasible problem"),
        }
    }

    #[test]
    fn test_quadratic_problem_scaling() {
        // Test that the solver handles different scales of quadratic problems
        variables! {
            vars:
                x;
                y;
        }

        // Large-scale problem: minimize 1000000*x^2 + 1000000*y^2
        let mut objective = QuadraticAffineExpression::new();
        objective.add_quadratic_term(x, x, 1_000_000.0);
        objective.add_quadratic_term(y, y, 1_000_000.0);

        let solution = vars
            .minimise_quadratic(objective)
            .using(clarabel_quadratic)
            .with(x + y >> 1e-3) // Very small constraint
            .solve()
            .expect("Scaled quadratic should solve");

        // Should still find optimal solution near x=y=5e-4
        assert!((solution.value(x) - 5e-4).abs() < 1e-6);
        assert!((solution.value(y) - 5e-4).abs() < 1e-6);
    }
}

// Provide a message when quadratic features are not enabled
#[cfg(not(all(feature = "clarabel", feature = "enable_quadratic")))]
#[test]
fn quadratic_features_not_enabled() {
    println!("Quadratic programming tests require both 'clarabel' and 'enable_quadratic' features");
    println!("Run with: cargo test --features 'clarabel,enable_quadratic'");
}
