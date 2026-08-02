//! A complete example that shows how to use `good_lp` **without** the
//! [`variables!`](good_lp::variables) and [`constraint!`](good_lp::constraint) macros.
//!
//! This example solves the very same [resource allocation problem](https://github.com/rust-or/good_lp/blob/main/tests/resource_allocation_problem.rs)
//! as the main documentation, but builds every part of the model using the
//! plain struct and method API provided by the crate:
//!
//! - [`ProblemVariables::new`] and [`ProblemVariables::add`] instead of [`variables!`]
//! - [`Expression`]s built with standard arithmetic on [`Variable`]s and `Iterator::sum`
//! - [`Expression::leq`], e.g. `.leq(available)`, and [`SolverModel::add_constraint`]
//!   instead of [`constraint!`]
//!
//! This is the way to go whenever the size of the problem, or its set of
//! variables and constraints, is only known at run time (generated from data,
//! loaded from a file, built in a loop, ...).
//!
//! We decide how much of two products to produce. Each unit of a product
//! consumes a fixed amount of fuel and time, and yields a profit. We want to
//! maximise the total profit without exceeding the available fuel and time.

use std::error::Error;

use float_eq::assert_float_eq;
use good_lp::{
    Expression, ProblemVariables, Solution, SolverModel, Variable, default_solver, variable,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

/// A product we can produce and sell. Producing one unit consumes `needed_fuel`
/// fuel and `needed_time` time, and brings a `value` profit.
struct Product {
    needed_fuel: f64,
    needed_time: f64,
    value: f64,
}

/// The total amount of a resource consumed by producing the given amounts.
fn resource_consumed<'a, F>(
    products: impl IntoIterator<Item = (&'a Product, &'a Variable)>,
    consumption: F,
) -> Expression
where
    F: Fn(&Product) -> f64,
{
    products
        .into_iter()
        .map(|(product, amount)| consumption(product) * *amount)
        .sum()
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn no_macro_example() -> Result<(), Box<dyn Error>> {
    // 1. Declare the resources we have in limited quantity.
    let available_fuel = 5.;
    let available_time = 3.;

    // 2. Declare the products we can produce. The number of products can be
    //    arbitrary: it could come from a file, a database, a loop, ...
    let products = [
        Product {
            needed_fuel: 1.,
            needed_time: 1.,
            value: 10.,
        },
        Product {
            needed_fuel: 2.,
            needed_time: 1.,
            value: 11.,
        },
    ];

    // 3. Create the set of variables without the variables!{} macro.
    //    For each product, the variable is the (non-negative) quantity to produce.
    let mut vars = ProblemVariables::new();
    let amounts: Vec<Variable> = products
        .iter()
        .map(|_| vars.add(variable().min(0)))
        .collect();

    // 4. Build the objective function (total profit) as an Expression, adding
    //    one term per product.
    let objective: Expression = products
        .iter()
        .zip(amounts.iter())
        .map(|(p, amount)| p.value * *amount)
        .sum();

    // 5. Create the model and add the resource constraints with
    //    SolverModel::add_constraint, using Expression::leq.
    let mut model = vars.maximise(objective.clone()).using(default_solver);
    model.add_constraint(
        resource_consumed(products.iter().zip(&amounts), |p| p.needed_fuel).leq(available_fuel),
    );
    model.add_constraint(
        resource_consumed(products.iter().zip(&amounts), |p| p.needed_time).leq(available_time),
    );

    // 6. Solve the problem and print the result.
    let solution = model.solve()?;
    println!(
        "produce {:.0} units of product 1 and {:.0} units of product 2",
        solution.value(amounts[0]),
        solution.value(amounts[1])
    );
    println!("total profit: {}", solution.eval(&objective));

    // 7. Check that the solution is the expected one.
    assert_float_eq!(solution.value(amounts[0]), 1., abs <= 1e-8);
    assert_float_eq!(solution.value(amounts[1]), 2., abs <= 1e-8);
    Ok(())
}
