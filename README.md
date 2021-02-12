# good_lp

A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.

```rust
use good_lp::{variables, coin_cbc, SolverModel, Solution};

fn main() {
    let mut vars = variables!();
    let a = vars.add_variable();
    let b = vars.add_variable();
    let solution = vars.maximise(9. * (a * 2 + b / 3))
        .using(coin_cbc)
        .with(a + 2. << b)
        .with(3. - a >> b)
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
}
```