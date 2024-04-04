# General guidelines

Contribution happens through github.

Please format your code with `cargo fmt`, and if you add a new feature, update the README, the doc comments, and the doc tests accordingly.

# Adding a new solver

Adding a new solver should not take more than a few hundred lines of code, tests included.

 - add the solver as an optional dependency in `Cargo.toml`
 - add a file named after your solver in the [solvers](./src/solvers) folder
    - you can copy minilp, our smallest solver interface, as a starting point.
    - create a struct to store linear problems in a way that will make it cheap to dynamically add new constraints to the problem,
       and easy to pass the problem to the solver once it has been fully constructed.
      This generally means constructing vectors to which you can push values for each new constraint.
       You can generally reuse data structures provided by the library for which you are creating a wrapper. 
    - implement a function named after your solver that takes an [`UnsolvedProblem`](https://docs.rs/good_lp/latest/good_lp/variable/struct.UnsolvedProblem.html) and returns the struct you defined above.
    - implement the [`SolverModel`](https://docs.rs/good_lp/latest/good_lp/index.html#reexport.SolverModel) trait for your new problem type.
    - add your solver to `lib.rs` and to the `all_default_solvers` feature in Cargo.toml. 
    - open a [pull request](https://github.com/rust-or/good_lp/pulls)
