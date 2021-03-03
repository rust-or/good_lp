#[allow(clippy::float_cmp)]
#[cfg(feature = "highs")]
mod tests {
    use good_lp::{
        constraint, dual::DualValues, highs, solvers::highs::HighsSolution, variable, variables,
        Solution, SolverModel,
    };

    #[test]
    fn determine_shadow_prices() {
        let mut vars = variables!();
        // Non-negative values
        let x1 = vars.add(variable().min(0));
        let x2 = vars.add(variable().min(0));

        // Subject to
        let c1 = constraint!(4.44 * x1 <= 100.0);
        let c2 = constraint!(6.67 * x2 <= 100.0);
        let c3 = constraint!(4 * x1 + 2.86 * x2 <= 100.0);
        let c4 = constraint!(3 * x1 + 6 * x2 <= 100.0);

        // Objective
        let objective = 3 * x1 + 2.5 * x2;

        let solution: HighsSolution = vars
            .maximise(objective.clone())
            .using(highs)
            .with(c1)
            .with(c2)
            .with(c3)
            .with(c4)
            .solve()
            .expect("Library test");

        assert_eq!(solution.eval(&objective), 77.30220492866408);
        assert_eq!(solution.value(x1), 20.363164721141374);
        assert_eq!(solution.value(x2), 6.485084306095981);
        assert_eq!(
            solution.get_dual_values(),
            vec![-0.0, -0.0, -0.6809338521400778, -0.09208819714656294]
        );
    }

    #[test]
    fn furniture_problem() {
        let mut vars = variables!();
        // Non-negative values
        let n_chairs = vars.add(variable().min(0));
        let n_tables = vars.add(variable().min(0));

        // Subject to
        let carpentry_constraint = constraint!(4 * n_chairs + 3 * n_tables <= 240.0);
        let painting_constraint = constraint!(2 * n_chairs + n_tables <= 100.0);

        // Objective
        let objective = 70 * n_chairs + 50 * n_tables;

        let solution: HighsSolution = vars
            .maximise(objective.clone())
            .using(highs)
            .with(carpentry_constraint)
            .with(painting_constraint)
            .solve()
            .expect("Library test");

        assert_eq!(solution.eval(&objective) as u32, 4100);
        assert_eq!(solution.value(n_chairs) as u32, 29);
        assert_eq!(solution.value(n_tables) as u32, 40);
        assert_eq!(
            solution.get_dual_values(),
            vec![-15.000000000000004, -4.999999999999992]
        );
    }
}
