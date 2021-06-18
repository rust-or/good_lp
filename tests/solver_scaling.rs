use float_eq::assert_float_eq;
use good_lp::{constraint, default_solver, variable, variables, Expression, Solution, SolverModel};

const BIG_NUM: usize = 1000; // <- Set this higher to test how good_lp and the solvers scale

#[test]
fn solve_large_problem() {
    let mut vars = variables!();
    let min = -((BIG_NUM / 2) as f64);
    let max = (BIG_NUM / 2 - 1) as f64;
    let v = vars.add_vector(variable().min(min).max(max), BIG_NUM);
    let objective: Expression = v.iter().sum();
    let mut pb = vars.maximise(objective).using(default_solver);
    for vs in v.windows(2) {
        pb = pb.with(constraint!(vs[0] + 1 <= vs[1]));
    }
    let sol = pb.solve().unwrap();
    for (i, var) in v.iter().enumerate() {
        assert_float_eq!(sol.value(*var), min + i as f64, abs <= 1e-10);
    }
}

#[test]
fn add_10_000_constraints() {
    let mut vars = variables!();
    let v = vars.add_vector(variable(), 10_000);
    let mut pb = vars.maximise(v[0]).using(default_solver);
    for vs in v.windows(2) {
        pb = pb.with(constraint!(vs[0] + 1 <= vs[1]));
    }
}

#[test]
fn sum_binaries() {
    // See: https://github.com/rust-or/good_lp/issues/8
    let mut vars = variables!();
    let team1_bools = vars.add_vector(variable().binary(), BIG_NUM);
    let team1_score: Expression = team1_bools.iter().sum();
    let _constraint = constraint!(team1_score == 5);
}
