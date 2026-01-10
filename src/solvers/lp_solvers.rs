//! This module allows solving problems is external solver binaries.
//! Contrarily to other solver modules, this one doesn't require linking your program to any solver.
//! A solver binary will need to be present on the user's computer at runtime.

use std::cmp::Ordering;
use std::collections::HashMap;

use lp_solvers::lp_format::LpObjective;
use lp_solvers::problem::StrExpression;
pub use lp_solvers::solvers::*;
use lp_solvers::util::UniqueNameGenerator;

use crate::constraint::ConstraintReference;
use crate::solvers::{MipGapError, ObjectiveDirection, SolutionStatus, WithTimeLimit};
use crate::variable::UnsolvedProblem;
use crate::{
    Constraint, Expression, IntoAffineExpression, ResolutionError, Solution as GoodLpSolution,
    Solver, SolverModel, Variable,
};

/// An external solver
pub struct LpSolver<T: lp_solvers::solvers::SolverTrait>(pub T);

impl<T: lp_solvers::solvers::SolverTrait + Clone> Solver for LpSolver<T> {
    type Model = Model<T>;

    fn create_model(&mut self, problem: UnsolvedProblem) -> Self::Model {
        let name = "good_lp_problem".to_string();
        let sense = match problem.direction {
            ObjectiveDirection::Maximisation => LpObjective::Maximize,
            ObjectiveDirection::Minimisation => LpObjective::Minimize,
        };
        let mut gen = UniqueNameGenerator::default();
        let variables: Vec<lp_solvers::problem::Variable> = problem
            .variables
            .into_iter()
            .map(|def| lp_solvers::problem::Variable {
                name: gen.add_variable(&def.name).to_string(),
                is_integer: def.is_integer,
                lower_bound: def.min,
                upper_bound: def.max,
            })
            .collect();
        let objective = linear_coefficients_str(&problem.objective, &variables);
        Model {
            problem: lp_solvers::problem::Problem {
                name,
                sense,
                objective,
                variables,
                constraints: vec![],
            },
            solver: self.0.clone(),
        }
    }

    fn name() -> &'static str {
        <Model<T> as SolverModel>::name()
    }
}

impl<T> crate::solvers::WithMipGap for Model<T>
where
    T: lp_solvers::solvers::WithMipGap<T>,
{
    fn mip_gap(&self) -> Option<f32> {
        self.solver.mip_gap()
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        match self.solver.with_mip_gap(mip_gap) {
            Ok(solver) => {
                self.solver = solver;
                Ok(self)
            }
            Err(err) => {
                if mip_gap.is_sign_negative() {
                    Err(MipGapError::Negative)
                } else if mip_gap.is_infinite() {
                    Err(MipGapError::Infinite)
                } else {
                    Err(MipGapError::Other(err))
                }
            }
        }
    }
}

/// A problem to be used by lp-solvers
pub struct Model<T> {
    problem: lp_solvers::problem::Problem,
    solver: T,
}

impl<T: SolverTrait> SolverModel for Model<T> {
    type Solution = LpSolution;
    type Error = ResolutionError;

    fn solve(self) -> Result<Self::Solution, Self::Error> {
        let map = self.solver.run(&self.problem)?;
        match map.status {
            Status::Infeasible => return Err(ResolutionError::Infeasible),
            Status::Unbounded => return Err(ResolutionError::Unbounded),
            Status::NotSolved => return Err(ResolutionError::Other("unknown error: not solved")),
            _ => {}
        }
        let solution = self
            .problem
            .variables
            .iter()
            .map(|v| f64::from(*map.results.get(&v.name).unwrap_or(&0.)))
            .collect();
        Ok(LpSolution { solution })
    }

    fn add_constraint(&mut self, c: Constraint) -> ConstraintReference {
        let reference = ConstraintReference {
            index: self.problem.constraints.len(),
        };
        self.problem
            .constraints
            .push(lp_solvers::lp_format::Constraint {
                lhs: linear_coefficients_str(&c.expression, &self.problem.variables),
                operator: if c.is_equality {
                    Ordering::Equal
                } else {
                    Ordering::Less
                },
                rhs: -c.expression.constant,
            });
        reference
    }

    fn name() -> &'static str {
        "External Solver (through lp_solvers)"
    }
}

impl<T> crate::solvers::WithTimeLimit for Model<T>
where
    T: lp_solvers::solvers::WithMaxSeconds<T>,
{
    fn with_time_limit<U: Into<f64>>(mut self, seconds: U) -> Self {
        self.solver = self.solver.with_max_seconds(seconds.into() as u32);
        self
    }
}

fn linear_coefficients_str(
    expr: &Expression,
    variables: &[lp_solvers::problem::Variable],
) -> StrExpression {
    StrExpression(
        expr.linear_coefficients()
            .map(|(var, coeff)| format!("{:+} {}", coeff, variables[var.index()].name))
            .collect::<Vec<String>>()
            .join(" "),
    )
}

impl<T> crate::solvers::WithInitialSolution for Model<T>
where
    T: WithMipStart<T>,
{
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        let mut start: HashMap<String, f32> = HashMap::new();

        for (v, val) in solution {
            if !val.is_finite() {
                continue;
            }

            let idx = v.index();
            let Some(lp_var) = self.problem.variables.get(idx) else {
                continue;
            };

            let val_f32 = val as f32;
            if !val_f32.is_finite() {
                continue;
            }

            start.insert(lp_var.name.clone(), val_f32);
        }

        if let Ok(solver) = self.solver.with_mip_start(&start) {
            self.solver = solver;
        }

        self
    }
}

/// A solution
pub struct LpSolution {
    solution: Vec<f64>,
}

impl GoodLpSolution for LpSolution {
    fn status(&self) -> SolutionStatus {
        SolutionStatus::Optimal
    }
    fn value(&self, variable: Variable) -> f64 {
        self.solution[variable.index()]
    }
}

#[cfg(test)]
mod tests {
    use crate::solvers::lp_solvers::{GlpkSolver, LpSolver};
    use crate::variables;

    #[test]
    fn coefficient_formatting_pos_pos() {
        variables! {vars: a; b; }
        let problem = vars
            .minimise(1 * a + 2 * b)
            .using(LpSolver(GlpkSolver::new()));
        assert_eq!(problem.problem.objective.0, "+2 b +1 a");
    }

    #[test]
    fn coefficient_formatting_pos_neg() {
        variables! {vars: a; b; }
        let problem = vars
            .minimise(1 * a - 2 * b)
            .using(LpSolver(GlpkSolver::new()));
        assert_eq!(problem.problem.objective.0, "-2 b +1 a");
    }

    #[test]
    fn coefficient_formatting_neg_pos() {
        variables! {vars: a; b; }
        let problem = vars
            .minimise(-1 * a + 2 * b)
            .using(LpSolver(GlpkSolver::new()));
        assert_eq!(problem.problem.objective.0, "+2 b -1 a");
    }

    #[test]
    fn coefficient_formatting_neg_neg() {
        variables! {vars: a; b; }
        let problem = vars
            .minimise(-1 * a - 2 * b)
            .using(LpSolver(GlpkSolver::new()));
        assert_eq!(problem.problem.objective.0, "-2 b -1 a");
    }
}
