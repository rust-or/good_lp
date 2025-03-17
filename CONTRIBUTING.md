# General Guidelines

Contribution happens through GitHub.

Please format your code with `cargo fmt`, and if you add a new feature, update the README, the doc comments, and the doc tests accordingly.

## Adding a New Solver

Adding a new solver should not take more than a few hundred lines of code, tests included.

- add the solver as an optional dependency in `Cargo.toml`
- add a file named after your solver in the [solvers](./src/solvers) folder
  - you can copy microlp, our smallest solver interface, as a starting point.
  - create a struct to store linear problems in a way that will make it cheap to dynamically add new constraints to the problem,
    and easy to pass the problem to the solver once it has been fully constructed.
    This generally means constructing vectors to which you can push values for each new constraint.
    You can generally reuse data structures provided by the library for which you are creating a wrapper.
  - implement a function named after your solver that takes an [`UnsolvedProblem`](https://docs.rs/good_lp/latest/good_lp/variable/struct.UnsolvedProblem.html) and returns the struct you defined above.
  - implement the [`SolverModel`](https://docs.rs/good_lp/latest/good_lp/index.html#reexport.SolverModel) trait for your new problem type.
- add your solver to `lib.rs` and to the `all_default_solvers` feature in Cargo.toml.
- run the tests of your solver [in the CI setup]([url](https://github.com/rust-or/good_lp/blob/main/.github/workflows/rust.yml))
- open a [pull request](https://github.com/rust-or/good_lp/pulls)

## Dev Container Setup

This repository contains a dev container definition.
This gives you a working system setup with all solvers installed with a single command.
It will also give you the entire Rust toolchain as well as all necessary VS Code extensions.

1. Make sure you have [Docker](https://docs.docker.com/engine/install/) installed on your system
2. Make sure you have the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed in VS Code
3. Run the command `Dev Containers: Reopen in Container` in VS Code

Done.
