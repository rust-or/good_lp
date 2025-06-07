use good_lp::{variables, Expression};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn complex_expression() {
    let mut var1 = variables!();
    let a = var1.add_variable();
    let b = var1.add_variable();
    let c = var1.add_variable();
    let d = var1.add_variable();
    assert_eq!(
        9. * (a - b * 2.) + 4 * c / 2 - d,
        9. * a + (-18.) * b + 2. * c + (-1.) * d
    )
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn large_sum() {
    let mut var1 = variables!();
    let var_vec: Vec<_> = (0..100_000).map(|_i| var1.add_variable()).collect();
    let sum_right: Expression = var_vec.iter().sum();
    let sum_reverse: Expression = var_vec.iter().rev().sum();
    assert_eq!(sum_right, sum_reverse)
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn debug_format() {
    let mut vars = variables!();
    let a = vars.add_variable();
    let b = vars.add_variable();
    let expr_str = format!("{:?}", (9 * (1. + a + b / 3)).leq(a + 1));
    let possibilities = vec!["3 v1 + 8 v0 <= -8", "8 v0 + 3 v1 <= -8"];
    assert!(
        possibilities.contains(&expr_str.as_str()),
        "expected one of {:?}, got {}",
        possibilities,
        expr_str
    )
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg(not(feature = "clarabel"))]
fn variables_macro_integer() {
    use good_lp::{constraint, default_solver, variables, Solution, SolverModel};

    variables! {
        vars:
               a <= 1;
          2 <= b (integer) <= 4;
    }
    let solution = vars
        .maximise(10 * (a - b / 5) - b)
        .using(default_solver)
        .with(constraint!(a + 2 <= b))
        .with(constraint!(1 + a >= 4 - b))
        .solve()
        .expect("solve");
    assert!((solution.value(a) - 1.).abs() < 1e-5);
    assert!((solution.value(b) - 3.).abs() < 1e-5);
}
