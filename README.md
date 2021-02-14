# good_lp

A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.

[![documentation](https://docs.rs/good_lp/badge.svg)](https://docs.rs/good_lp)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

```rust
use good_lp::{variables, variable, coin_cbc, SolverModel, Solution};

fn main() {
    let mut vars = variables!();
    let a = vars.add(variable().max(1));
    let b = vars.add(variable().min(2).max(4));
    let solution = vars.maximise(10 * (a - b / 5) - b)
        .using(coin_cbc)
        .with(a + 2 << b) // or (a + 2).leq(b)
        .with(1 + a >> 4 - b) // or (1 + a).geq(4 - b)
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    println!("a + b = {}", solution.eval(a + b));
}
```

## Features and limitations

 - **Linear programming**. This crate currently supports only the definition
   of linear programs. You cannot use it with quadratic functions, for instance.
 - **Continuous variables**. Currently, only continuous variables are supported.
   Mixed-integer linear programming (MILP) is not supported.
 - **Not a solver**. This crate uses other rust crates to provide the solvers.
   There is no solving algorithm in good_lp itself. If you have an issue with a solver,
   report it to the solver directly. See below for the list of supported solvers.

### Contributing

Pull requests are welcome !
If you need any of the features mentioned above, get in touch.
Also, do not hesitate to open issues to discuss the implementation.

## Usage examples

You can find a resource allocation problem example in
[`resource_allocation_problem.rs`](./tests/resource_allocation_problem.rs).

## Solvers

This library offers an abstraction over multiple solvers. By default, it uses [cbc](https://www.coin-or.org/Cbc/), but
you can also activate other solvers using cargo features.

#### [cbc](https://www.coin-or.org/Cbc/)

Used by default, performant, but requires to have a C compiler and the cbc C library installed.

In ubuntu, you can install it with:

```
sudo apt-get install coinor-cbc coinor-libcbc-dev
```

In MacOS, using [homebrew](https://brew.sh/) :

```
brew install cbc
```

#### [minilp](https://docs.rs/minilp)

minilp is a pure rust solver, which means it works out of the box without installing anything else.

You can activate it with :

```toml
[dependencies.good_lp]
version = "0.1.0"
default-features = false
features = ["minilp"]
```

Then use `minilp` instead of `coin_cbc` in your code:
```rust
use good_lp::minilp;

fn optimize<V>(vars: ProblemVariables<V>) {
    vars.maximise(objective).using(minilp);
}
```

### License

This library is published under the MIT license.