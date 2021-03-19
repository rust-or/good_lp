#[cfg(feature = "highs")]
mod dual_tests {
    use float_eq::assert_float_eq;

    use good_lp::{
        constraint, highs,
        solvers::{highs::HighsSolution, DualValues, SolutionWithDual},
        variable, variables, Solution, SolverModel,
    };

    #[test]
    fn determine_shadow_prices() {
        // Instantiate the Variables
        let mut vars = variables!();
        // Non-negative values
        let x1 = vars.add(variable().min(0));
        let x2 = vars.add(variable().min(0));

        // Define the Problem and Objective
        let objective = 3 * x1 + 2 * x2;
        let mut p = vars.maximise(objective.clone()).using(highs);

        // Subject to
        let c1 = p.add_constraint(constraint!(4 * x1 <= 100.0));
        let c2 = p.add_constraint(constraint!(7 * x2 <= 100.0));
        let c3 = p.add_constraint(constraint!(4 * x1 + 3 * x2 <= 100.0));
        let c4 = p.add_constraint(constraint!(3 * x1 + 6 * x2 <= 100.0));

        // Solve Problem
        let mut solution = p.solve().expect("Library test");

        assert_float_eq!(75.0, solution.eval(&objective), abs <= 1e-3);
        assert_float_eq!(25.0, solution.value(x1), abs <= 1e-3);
        assert_float_eq!(-0.0, solution.value(x2), abs <= 1e-3);

        let dual = solution.compute_dual();
        assert_float_eq!(0., dual.dual(c1), abs <= 1e-1);
        assert_float_eq!(0., dual.dual(c2), abs <= 1e-1);
        assert_float_eq!(-0.667, dual.dual(c3), abs <= 1e-3);
        assert_float_eq!(-0.0, dual.dual(c4), abs <= 1e-3);
    }

    #[test]
    fn furniture_problem() {
        // Non-negative values
        let mut vars = variables!();
        let n_chairs = vars.add(variable().min(0));
        let n_tables = vars.add(variable().min(0));

        // Objective and Problem
        let objective = 70 * n_chairs + 50 * n_tables;
        let mut p = vars.maximise(objective.clone()).using(highs);

        // Subject to
        let c1 = p.add_constraint(constraint!(4 * n_chairs + 3 * n_tables <= 240.0));
        let c2 = p.add_constraint(constraint!(2 * n_chairs + n_tables <= 100.0));

        // Solve
        let mut solution: HighsSolution = p.solve().expect("Library test");
        assert_float_eq!(4100.0, solution.eval(&objective), abs <= 1e-10);
        assert_float_eq!(30.0, solution.value(n_chairs), abs <= 1e-1);
        assert_float_eq!(40.0, solution.value(n_tables), abs <= 1e-1);

        let dual = solution.compute_dual();
        assert_float_eq!(-15.0, dual.dual(c1), abs <= 1e-1);
        assert_float_eq!(-5.0, dual.dual(c2), abs <= 1e-1);
    }
}
