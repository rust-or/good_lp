//! In this example, we have multiple products,
//! and each consumes a set amount of fuel, and of time to produce.
//! The goal is to find, knowing the available fuel and time,
//! and the value of each product, how much we should produce of each.

use good_lp::variable::ProblemVariables;
use good_lp::{
    coin_cbc, variable, variables, Expression, Solution, SolverModel, Variable,
};

struct ResourceAllocationProblem<F> {
    vars: ProblemVariables<F>,
    total_value: Expression<F>,
    consumed_fuel: Expression<F>,
    consumed_time: Expression<F>,
    available_fuel: f64,
    available_time: f64,
}

impl<F> ResourceAllocationProblem<F> {
    fn new(
        variables: ProblemVariables<F>,
        available_fuel: f64,
        available_time: f64,
    ) -> ResourceAllocationProblem<F> {
        ResourceAllocationProblem {
            vars: variables,
            available_fuel,
            available_time,
            consumed_fuel: 0.into(),
            consumed_time: 0.into(),
            total_value: 0.into(),
        }
    }
    fn add_product(&mut self, needed_fuel: f64, needed_time: f64, value: f64) -> Variable<F> {
        let product = self.vars.add(variable().min(0));
        self.total_value += product * value;
        self.consumed_fuel += product * needed_fuel;
        self.consumed_time += product * needed_time;
        product
    }

    fn best_product_quantities(self) -> impl Solution<F> {
        self.vars
            .maximise(self.total_value)
            .using(coin_cbc)
            .with(self.consumed_fuel.leq(self.available_fuel))
            .with(self.consumed_time.leq(self.available_time))
            .solve()
            .unwrap()
    }
}

#[test]
fn resource_allocation() {
    let mut pb = ResourceAllocationProblem::new(variables!(), 5., 3.);
    let steel = pb.add_product(1., 1., 10.);
    let stainless_steel = pb.add_product(2., 1., 11.);

    let solution = pb.best_product_quantities();

    // The amount of steel we should produce
    assert_eq!(1., solution.value(steel));
    // The amount of stainless steel we should produce
    assert_eq!(2., solution.value(stainless_steel));
}
