use std::error::Error;

use good_lp::{Solution, SolverModel, constraint, default_solver, variables};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn main() -> Result<(), Box<dyn Error>> {
    variables! {
        vars:
               a <= 1;
          2 <= b <= 4;
    }
    let solution = vars
        .maximise(10 * (a - b / 5) - b)
        .using(default_solver)
        .with(constraint!(a + 2 <= b))
        .with(constraint!(1 + a >= 4 - b))
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    println!("a + b = {}", solution.eval(a + b));
    Ok(())
}
