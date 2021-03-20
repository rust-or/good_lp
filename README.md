# good_lp

A Linear Programming modeler that is easy to use, performant with large problems, and well-typed.

[![Crates.io](https://img.shields.io/crates/v/good_lp.svg)](https://crates.io/crates/good_lp)
[![documentation](https://docs.rs/good_lp/badge.svg)](https://docs.rs/good_lp)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

```rust
use good_lp::{variables, variable, coin_cbc, SolverModel, Solution, contraint};

fn main() {
    variables!{
        vars:
               a <= 1;
          2 <= b <= 4;
    };
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
  you can maximise `3 * x + y`, but not `3 * x * y`.
- **Continuous and integer variables**. good_lp supports mixed integer-linear programming (MILP),
  but not all underlying solvers support integer variables.
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

This library offers an abstraction over multiple solvers. By default, it uses [cbc][cbc], but
you can also activate other solvers using cargo features.

| solver feature name | integer variables | no C compiler\*| no additional libs\** | fast |
|---------------------|---------------|-------------------|------------------------|------|
| [`coin_cbc`][cbc]   | ✅                 | ✅             | ❌                    | ✅
| [`highs`][highs]    | ❌                 | ❌             | ✅                    | ✅
| [`lpsolve`][lpsolve]| ✅                 | ❌             | ✅                    | ❌
| [`minilp`][minilp]  | ❌                 | ✅             | ✅                    | ❌

 * \* no C compiler: builds with only cargo, without requiring you to install a C compiler
 * \** no additional libs: works without additional libraries at runtime, all the dependencies are statically linked

To use an alternative solver, put the following in your `Cargo.toml`:

```toml
good_lp = { version = "*", features = ["your solver feature name"], default-features = false }
```


### [cbc][cbc]
Used by default, performant, but requires to have the cbc C library headers available on the build machine,
and the cbc dynamic library available on any machine where you want to run your program.

In ubuntu, you can install it with:

```bash
sudo apt-get install coinor-cbc coinor-libcbc-dev
```

In MacOS, using [homebrew](https://brew.sh/) :

```bash
brew install cbc
```

[cbc]: https://www.coin-or.org/Cbc/


### [minilp](https://docs.rs/minilp)

minilp is a pure rust solver, which means it works out of the box without installing anything else.

[minilp]: https://docs.rs/minilp

Minilp is written in pure rust, and performs poorly when compiled in debug mode. Be sure to compile your code
in `--release` mode when solving large problems.

### [lpsolve][lpsolve]

lp_solve is a free ([LGPL](http://lpsolve.sourceforge.net/5.5/LGPL.htm)) linear (integer) programming solver
written in C and based on the revised simplex method.

good_lp uses the [lpsolve crate](https://docs.rs/lpsolve/) to call lpsolve.
You will need a C compiler, but you won't have to install any additional library.

[lpsolve]:http://lpsolve.sourceforge.net/5.5/

### [HiGHS][highs]

HiGHS is a free ([MIT](https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE)) parallel linear programming solver
written in C++. It currently **doesn't support integer variables**.
It is able to use [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to fully leverage all the available processor cores
to solve a problem.

good_lp uses the [highs crate](https://docs.rs/highs) to call HiGHS.
You will need a C compiler, but you shouldn't have to install any additional library on linux
(it depends only on OpenMP and the C++ standard library).
More information in the [highs-sys crate](https://crates.io/crates/highs-sys).

[highs]: https://highs.dev

### License

This library is published under the MIT license.
The solver themselves have various licenses, please refer to their individual documentation.
