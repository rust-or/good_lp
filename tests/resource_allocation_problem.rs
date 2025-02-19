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
use good_lp::{
    constraint, default_solver, variable, variables, Constraint, Expression, Solution, SolverModel,
    Variable,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;
struct Product {
    // amount of fuel producing 1 unit takes
    needed_fuel: f64,
    // time it takes to produce 1 unit
    needed_time: f64,
    value: f64, // The amount of money we can sell an unit of the product for
}

/// Fuel and time are both resources, with a fixed amount available and a variable amount consumed by each product
struct Resource {
    available: f64,
    consumed: Expression,
}

struct ResourceSet {
    fuel: Resource,
    time: Resource,
}

struct ResourceAllocationProblem {
    /// Stores the variables of the problem
    vars: ProblemVariables,
    /// The total amount of money we can make by producing the products
    total_value: Expression,
    /// The resources available and consumed by the products
    resources: ResourceSet,
}

impl ResourceAllocationProblem {
    /// Create a new problem, with a fixed amount of fuel and time available
    fn new(available_fuel: f64, available_time: f64) -> ResourceAllocationProblem {
        ResourceAllocationProblem {
            vars: variables!(),
            total_value: 0.into(),
            resources: ResourceSet {
                fuel: Resource {
                    available: available_fuel,
                    consumed: 0.into(),
                },
                time: Resource {
                    available: available_time,
                    consumed: 0.into(),
                },
            },
        }
    }

    /// Add a new product to take into account in the optimization
    fn add(&mut self, product: Product) -> Variable {
        let amount_to_produce = self.vars.add(variable().min(0));
        self.total_value += amount_to_produce * product.value;
        self.resources.fuel.consumed += amount_to_produce * product.needed_fuel;
        self.resources.time.consumed += amount_to_produce * product.needed_time;
        amount_to_produce
    }

    /// Generate a vector containing the constraints.
    /// In this simple problem, our only constraints are that we can't consume more fuel or time than we have available.
    fn constraints(resources: ResourceSet) -> Vec<Constraint> {
        let mut constraints = Vec::with_capacity(2);
        for resource in [resources.fuel, resources.time] {
            constraints.push(constraint!(resource.consumed <= resource.available));
        }
        constraints
    }

    /// Solve the problem, returning the optimal amount of each product to produce.
    fn best_product_quantities(self) -> impl Solution {
        let objective = self.total_value;
        self.vars
            .maximise(objective)
            .using(default_solver)
            .with_all(Self::constraints(self.resources))
            .solve()
            .unwrap()
    }
}

use float_eq::assert_float_eq;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn resource_allocation() {
    let mut pb = ResourceAllocationProblem::new(5., 3.);
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
    assert_float_eq!(1., solution.value(steel), abs <= 1e-8);
    // The amount of stainless steel we should produce
    assert_float_eq!(2., solution.value(stainless_steel), abs <= 1e-8);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
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

    let mut pb = ResourceAllocationProblem::new(5., 3.);
    let variables: Vec<_> = products.into_iter().map(|p| pb.add(p)).collect();
    let solution = pb.best_product_quantities();
    let product_quantities: Vec<_> = variables.iter().map(|&v| solution.value(v)).collect();
    assert_float_eq!(1., product_quantities[0], abs <= 1e-8);
    assert_float_eq!(2., product_quantities[1], abs <= 1e-8);
}
