use float_eq::assert_float_eq;

use good_lp::{
    Solution, Solver, SolverModel, constraint,
    solvers::{DualValues, SolutionWithDual},
    variable, variables,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;
// Using a generic function here ensures the dual can be retrieved in a generic,
// solver-independent manner.
#[allow(dead_code)]
fn determine_shadow_prices_for_solver<S: Solver>(solver: S)
where
    for<'a> <<S as Solver>::Model as SolverModel>::Solution: SolutionWithDual<'a>,
{
    // Instantiate the Variables
    let mut vars = variables!();
    // Non-negative values
    let x1 = vars.add(variable().min(0));
    let x2 = vars.add(variable().min(0));

    // Define the Problem and Objective
    let objective = 3 * x1 + 2 * x2;
    let mut p = vars.maximise(objective.clone()).using(solver);

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
    assert_float_eq!(0.667, dual.dual(c3), abs <= 1e-1);
    assert_float_eq!(0.0, dual.dual(c4), abs <= 1e-3);
}

#[allow(dead_code)]
fn furniture_problem_for_solver<S: Solver>(solver: S)
where
    for<'a> <<S as Solver>::Model as SolverModel>::Solution: SolutionWithDual<'a>,
{
    // Non-negative values
    let mut vars = variables!();
    let n_chairs = vars.add(variable().min(0));
    let n_tables = vars.add(variable().min(0));

    // Objective and Problem
    let objective = 70 * n_chairs + 50 * n_tables;
    let mut p = vars.maximise(objective.clone()).using(solver);

    // Subject to
    let c1 = p.add_constraint(constraint!(4 * n_chairs + 3 * n_tables <= 240.0));
    let c2 = p.add_constraint(constraint!(2 * n_chairs + n_tables <= 100.0));

    // Solve
    let mut solution = p.solve().expect("Library test");
    assert_float_eq!(4100.0, solution.eval(&objective), abs <= 1e-5);
    assert_float_eq!(30.0, solution.value(n_chairs), abs <= 1e-1);
    assert_float_eq!(40.0, solution.value(n_tables), abs <= 1e-1);

    let dual = solution.compute_dual();
    assert_float_eq!(15.0, dual.dual(c1), abs <= 1e-1);
    assert_float_eq!(5.0, dual.dual(c2), abs <= 1e-1);
}

#[allow(dead_code)]
fn greater_than_shadow_price_for_solver<S: Solver>(solver: S)
where
    for<'a> <<S as Solver>::Model as SolverModel>::Solution: SolutionWithDual<'a>,
{
    let mut vars = variables!();
    let x = vars.add_vector(variable().min(0), 4);
    let objective = 4 * x[0] + 6 * x[1] + 7 * x[2] + 8 * x[3];
    let mut problem = vars.maximise(objective).using(solver);

    problem.add_constraint(constraint!(
        2 * x[0] + 3 * x[1] + 4 * x[2] + 7 * x[3] <= 4600
    ));
    problem.add_constraint(constraint!(
        3 * x[0] + 4 * x[1] + 5 * x[2] + 6 * x[3] <= 5000
    ));
    let lower_bound = problem.add_constraint(constraint!(x[3] >= 400));
    problem.add_constraint(constraint!(x[0] + x[1] + x[2] + x[3] == 950));

    let mut solution = problem.solve().expect("Library test");
    let dual = solution.compute_dual();

    // At the optimum x2 = x4 = 400. Increasing x4's lower bound by one
    // exchanges one unit of x2 (coefficient 6) for x4 (coefficient 8), a
    // change of 8 - 6 = 2; lower-bounded row duals use the opposite sign.
    assert_float_eq!(-2.0, dual.dual(lower_bound), abs <= 1e-3);
}

#[allow(dead_code)]
fn minimization_greater_than_shadow_price_for_solver<S: Solver>(solver: S)
where
    for<'a> <<S as Solver>::Model as SolverModel>::Solution: SolutionWithDual<'a>,
{
    let mut vars = variables!();
    let x = vars.add(variable().min(0));
    let mut problem = vars.minimise(x).using(solver);
    let lower_bound = problem.add_constraint(constraint!(x >= 10));

    let mut solution = problem.solve().expect("Library test");
    assert_float_eq!(10.0, solution.value(x), abs <= 1e-3);

    // Raising the lower bound by one raises the minimum of x by one.
    let dual = solution.compute_dual();
    assert_float_eq!(1.0, dual.dual(lower_bound), abs <= 1e-3);
}

#[allow(dead_code)]
fn minimization_less_than_shadow_price_for_solver<S: Solver>(solver: S)
where
    for<'a> <<S as Solver>::Model as SolverModel>::Solution: SolutionWithDual<'a>,
{
    let mut vars = variables!();
    let x = vars.add(variable());
    let mut problem = vars.minimise(-x).using(solver);
    let upper_bound = problem.add_constraint(constraint!(x <= 10));

    let mut solution = problem.solve().expect("Library test");
    assert_float_eq!(10.0, solution.value(x), abs <= 1e-3);

    // Raising the upper bound by one lowers the minimum of -x by one.
    let dual = solution.compute_dual();
    assert_float_eq!(-1.0, dual.dual(upper_bound), abs <= 1e-3);
}

macro_rules! dual_test {
    ($([$solver_feature:literal, $solver:expr])*) => {
        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn determine_shadow_prices() {
            $(
                #[cfg(feature = $solver_feature)]
                determine_shadow_prices_for_solver($solver);
            )*
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn furniture_problem() {
            $(
                #[cfg(feature = $solver_feature)]
                furniture_problem_for_solver($solver);
            )*
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn greater_than_shadow_price() {
            $(
                #[cfg(feature = $solver_feature)]
                greater_than_shadow_price_for_solver($solver);
            )*
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn greater_than_shadow_price_for_minimization() {
            $(
                #[cfg(feature = $solver_feature)]
                minimization_greater_than_shadow_price_for_solver($solver);
            )*
        }

        #[test]
        #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
        fn less_than_shadow_price_for_minimization() {
            $(
                #[cfg(feature = $solver_feature)]
                minimization_less_than_shadow_price_for_solver($solver);
            )*
        }
    };
}

dual_test!(
    ["highs", good_lp::highs]
    ["clarabel", good_lp::clarabel]
);
