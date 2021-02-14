use criterion::{black_box, criterion_group, criterion_main, Criterion};
use good_lp::{variables, Expression};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sum((2 x_i + 1) for i in [1..100_000])", |b| {
        b.iter(|| {
            let mut vars = variables!();
            let v: Expression<_> = (0..100_000)
                .map(|_i| {
                    let x_i = vars.add_variable();
                    black_box(2) * black_box(x_i) + black_box(1)
                })
                .sum();
            v
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
