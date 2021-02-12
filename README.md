# good_lp

A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.

```rust
use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};

fn main() {
    let mut vars = variables!();
    let a = vars.add(variable().max(1));
    let b = vars.add(variable().min(2).max(4));
    let solution = vars.maximise(10 * (a - b / 5) - b)
        .using(coin_cbc)
        .with(a + 2. << b)
        .with(1 + a >> 4. - b)
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    println!("a + b = {}", solution.eval(a + b));
}
```