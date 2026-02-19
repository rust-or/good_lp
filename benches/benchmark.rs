use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use good_lp::{Expression, Solution, SolverModel, default_solver, variable, variables};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sum((2 x_i + 1) for i in [1..100_000])", |b| {
        b.iter(|| {
            let mut vars = variables!();
            let v: Expression = (0..100_000)
                .map(|_i| {
                    let x_i = vars.add_variable();
                    black_box(2) * black_box(x_i) + black_box(1)
                })
                .sum();
            v
        })
    });

    c.bench_function(
        "solving empty problem with 1M variables and reading results",
        |b| {
            b.iter(|| {
                let mut vars = variables!();
                let vs = vars.add_vector(variable().min(0).name("test"), 1_000_000);
                let obj: Expression = vs.iter().sum();
                let sol = vars.minimise(&obj).using(default_solver).solve().unwrap();
                sol.eval(obj)
            })
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
