use good_lp::{variables, Expression};

#[test]
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
fn large_sum() {
    let mut var1 = variables!();
    let var_vec: Vec<_> = (0..100_000).map(|_i| var1.add_variable()).collect();
    let sum_right: Expression<_> = var_vec.iter().sum();
    let sum_reverse: Expression<_> = var_vec.iter().rev().sum();
    assert_eq!(sum_right, sum_reverse)
}

#[test]
fn complete() {
    let mut var1 = variables!();
    let mut var2 = variables!();
    assert_eq!(
        // variables iss the size of an empty vector
        std::mem::size_of_val(&Vec::<u8>::new()),
        std::mem::size_of_val(&var1)
    );
    let a = var1.add_variable();
    let b = var2.add_variable();
    let _sum_a = a + a;
    let _diff_b = b - b + b;
}

#[test]
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
fn expected_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/failures/*.rs");
}
