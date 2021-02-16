//! In this example, we have multiple products,
//! and each consumes a set amount of fuel, and of time to produce.
//! The goal is to find, knowing the available fuel and time,
//! and the value of each product, how much we should produce of each.
//!
//! In this example, the number of resources is fixed (only fuel an time),
//! and the amount of products varies.
//! In the opposite case (a fixed number of products and an arbitrary number of resources),
//! the modelling is even simpler: you don't have to store any expression in your problem struct,
//! you can instantiate a SolverModel directly when creating your problem,
//! and then use SolverModel::with to add constraints dynamically.

use good_lp::variable::ProblemVariables;
use good_lp::{default_solver, variable, variables, Expression, Solution, SolverModel, Variable};

struct Product {
    // amount of fuel producing 1 unit takes
    needed_fuel: f64,
    // time it takes to produce 1 unit
    needed_time: f64,
    value: f64, // The amount of money we can sell an unit of the product for
}

struct ResourceAllocationProblem {
    vars: ProblemVariables,
    total_value: Expression,
    consumed_fuel: Expression,
    consumed_time: Expression,
    available_fuel: f64,
    available_time: f64,
}

impl ResourceAllocationProblem {
    fn new(
        variables: ProblemVariables,
        available_fuel: f64,
        available_time: f64,
    ) -> ResourceAllocationProblem {
        ResourceAllocationProblem {
            vars: variables,
            available_fuel,
            available_time,
            consumed_fuel: 0.into(),
            consumed_time: 0.into(),
            total_value: 0.into(),
        }
    }

    /// Add a new product to take into account in the optimization
    fn add(&mut self, product: Product) -> Variable {
        let amount_to_produce = self.vars.add(variable().min(0));
        self.total_value += amount_to_produce * product.value;
        self.consumed_fuel += amount_to_produce * product.needed_fuel;
        self.consumed_time += amount_to_produce * product.needed_time;
        amount_to_produce
    }

    fn best_product_quantities(self) -> impl Solution {
        self.vars
            .maximise(self.total_value)
            .using(default_solver)
            .with(self.consumed_fuel.leq(self.available_fuel))
            .with(self.consumed_time.leq(self.available_time))
            .solve()
            .unwrap()
    }
}

#[test]
fn resource_allocation() {
    let mut pb = ResourceAllocationProblem::new(variables!(), 5., 3.);
    let steel = pb.add(Product {
        needed_fuel: 1.,
        needed_time: 1.,
        value: 10.,
    });
    let stainless_steel = pb.add(Product {
        needed_fuel: 2.,
        needed_time: 1.,
        value: 11.,
    });

    let solution = pb.best_product_quantities();

    // The amount of steel we should produce
    assert_eq!(1., solution.value(steel));
    // The amount of stainless steel we should produce
    assert_eq!(2., solution.value(stainless_steel));
}

#[test]
fn using_a_vector() {
    let products = vec![
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

    let mut pb = ResourceAllocationProblem::new(variables!(), 5., 3.);
    let variables: Vec<_> = products.into_iter().map(|p| pb.add(p)).collect();
    let solution = pb.best_product_quantities();
    let product_quantities: Vec<_> = variables.iter().map(|&v| solution.value(v)).collect();
    assert_eq!(vec![1., 2.], product_quantities);
}
