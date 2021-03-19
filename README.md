# good_lp

A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.

[![documentation](https://docs.rs/good_lp/badge.svg)](https://docs.rs/good_lp)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

```rust
use good_lp::{variables, variable, coin_cbc, SolverModel, Solution, contraint};

fn main() {
    let mut vars = variables!();
    let a = vars.add(variable().max(1));
    let b = vars.add(variable().min(2).max(4));
    let solution = vars.maximise(10 * (a - b / 5) - b)
        .using(coin_cbc)
        .with(constraint!(a + 2 <= b))
        .with(constraint!(1 + a >= 4 - b))
        .solve()?;
    println!("a={}   b={}", solution.value(a), solution.value(b));
    println!("a + b = {}", solution.eval(a + b));
}
```

## Features and limitations

- **Linear programming**. This crate currently supports only the definition of linear programs. You cannot use it with
  quadratic functions, for instance:
  you can maximise `y + 3 * y`, but not `3 * x * y`.
- **Continuous variables**. Currently, only continuous variables are supported.
  Mixed-integer linear programming (MILP) is not supported.
- **Not a solver**. This crate uses other rust crates to provide the solvers.
  There is no solving algorithm in good_lp itself. If you have an issue with a solver,
  report it to the solver directly. See below for the list of supported solvers.

### Contributing

Pull requests are welcome !
If you need any of the features mentioned above, get in touch.
Also, do not hesitate to open issues to discuss the implementation.

### Alternatives

If you need non-linear programming or integer variables, you can use
[lp-modeler](https://crates.io/crates/lp-modeler).
However, it is currently very slow with large problems.

You can also directly use the underlying solver libraries, such as
[coin_cbc](https://docs.rs/coin_cbc/) or
[minilp](https://crates.io/crates/minilp)
if you don't need a way to express your objective function and
constraints using an idiomatic rust syntax.

## Usage examples

You can find a resource allocation problem example in
[`resource_allocation_problem.rs`](https://github.com/lovasoa/good_lp/blob/main/tests/resource_allocation_problem.rs).

## Solvers

This library offers an abstraction over multiple solvers. By default, it uses [cbc](https://www.coin-or.org/Cbc/), but
you can also activate other solvers using cargo features.

### [cbc](https://www.coin-or.org/Cbc/)

Used by default, performant, but requires to have a C compiler and the cbc C library installed.

In ubuntu, you can install it with:

```bash
sudo apt-get install coinor-cbc coinor-libcbc-dev
```

In MacOS, using [homebrew](https://brew.sh/) :

```bash
brew install cbc
```

### [minilp](https://docs.rs/minilp)

minilp is a pure rust solver, which means it works out of the box without installing anything else.

You can activate it with :

```toml
good_lp = { version = "0.4", features = ["minilp"], default-features = false }
```

Then use `minilp` instead of `coin_cbc` in your code:

```rust
use good_lp::minilp;

fn optimize<V>(vars: ProblemVariables<V>) {
   vars.maximise(objective).using(minilp);
}
```

Minilp is written in pure rust, and performs poorly when compiled in debug mode. Be sure to compile your code
in `--release` mode when solving large problems.

### [lpsolve](http://lpsolve.sourceforge.net/5.5/)

lp_solve is a free ([LGPL](http://lpsolve.sourceforge.net/5.5/LGPL.htm)) linear (integer) programming solver
written in C and based on the revised simplex method.

```toml
good_lp = { version = "0.4", features = ["lpsolve"], default-features = false }
```

good_lp uses the [lpsolve crate](https://docs.rs/lpsolve/) to call lpsolve.
You will need a C compiler, but you won't have to install any additional library.

### [HiGHS](https://highs.dev)

HiGHS is a free ([MIT](https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE)) parallel linear programming solver
written in C++.
It is able to leverage [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to fully leverage all the available processor cores
to solve a problem.

```toml
good_lp = { version = "0.4", features = ["highs"], default-features = false }
```

good_lp uses the [highs crate](https://docs.rs/highs) to call HiGHS.
You will need a C compiler, but you shouldn't have to install any additional library on linux
(it depends only on OpenMP and the C++ standard library).
More information in the [highs-sys crate](https://crates.io/crates/highs-sys).

### License

This library is published under the MIT license.
