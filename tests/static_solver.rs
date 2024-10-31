use good_lp::{constraint, variables, Solution, SolverModel, StaticSolver};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;
// See: https://github.com/rust-or/good_lp/pull/5

fn generic_solve_example<S: StaticSolver>(solver: S) -> Result<(), Box<dyn std::error::Error>> {
    variables! {
        vars:
               a <= 1;
          2 <= b <= 4;
    };
    let solution = vars
        .maximise(10 * (a - b / 5) - b)
        .using(solver)
        .with(constraint!(a + 2 <= b))
        .with(constraint!(1 + a >= 4 - b))
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    Ok(())
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn concrete() {
    generic_solve_example(good_lp::default_solver).expect("solve")
}
