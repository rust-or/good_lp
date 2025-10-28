//! A solver that uses a [Cbc](https://www.coin-or.org/Cbc/) [native library binding](https://docs.rs/coin_cbc).
//! This solver is activated using the default `coin_cbc` feature.
//! You can disable it an enable another solver instead using cargo features.
use std::convert::TryInto;

use coin_cbc::{
    raw::{SecondaryStatus, Status},
    Col, Model, Sense, Solution as CbcSolution,
};

use crate::solvers::{
    MipGapError, ModelWithSOS1, SolutionStatus, WithInitialSolution, WithMipGap, WithTimeLimit,
};
use crate::variable::{UnsolvedProblem, VariableDefinition};
use crate::{
    constraint::ConstraintReference,
    solvers::{ObjectiveDirection, ResolutionError, Solution, SolverModel},
    IntoAffineExpression,
};
use crate::{Constraint, Variable};

/// The Cbc [COIN-OR](https://www.coin-or.org/) solver library.
/// To be passed to [`UnsolvedProblem::using`](crate::variable::UnsolvedProblem::using)
pub fn coin_cbc(to_solve: UnsolvedProblem) -> CoinCbcProblem {
    let UnsolvedProblem {
        objective,
        direction,
        variables,
    } = to_solve;
    let mut model = Model::default();
    let mut initial_solution = Vec::with_capacity(variables.initial_solution_len());
    let columns: Vec<Col> = variables
        .iter_variables_with_def()
        .map(
            |(
                var,
                &VariableDefinition {
                    min,
                    max,
                    initial,
                    is_integer,
                    ..
                },
            )| {
                let col = model.add_col();
                // Variables are created with a default min of 0
                model.set_col_lower(col, min);
                if max < f64::INFINITY {
                    model.set_col_upper(col, max)
                }
                if is_integer {
                    model.set_integer(col);
                }
                if let Some(val) = initial {
                    initial_solution.push((var, val));
                };
                col
            },
        )
        .collect();
    for (var, coeff) in objective.linear.coefficients.into_iter() {
        model.set_obj_coeff(columns[var.index()], coeff);
    }
    model.set_obj_sense(match direction {
        ObjectiveDirection::Maximisation => Sense::Maximize,
        ObjectiveDirection::Minimisation => Sense::Minimize,
    });
    let mut problem = CoinCbcProblem {
        model,
        columns,
        has_sos: false,
        mip_gap: None,
    };
    if !initial_solution.is_empty() {
        problem = problem.with_initial_solution(initial_solution);
    }
    problem
}

/// A coin-cbc model
#[derive(Clone, Default)]
pub struct CoinCbcProblem {
    model: Model,
    columns: Vec<Col>,
    has_sos: bool,
    mip_gap: Option<f32>,
}

impl CoinCbcProblem {
    /// Get the inner coin_cbc model
    pub fn as_inner(&self) -> &Model {
        &self.model
    }

    /// Get a mutable version of the inner Coin CBC model.
    /// good_lp will crash (but should stay memory-safe) if you change the structure of the problem
    /// itself using this method.
    pub fn as_inner_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Set an option in cbc. For the list of available options, start the cbc binary and type '?'
    /// ```
    /// use good_lp::*;
    /// variables!{ vars: 0<=x<=1; 0<=y<=1; }
    /// let mut model = vars.maximise(x + y).using(coin_cbc);
    /// model.set_parameter("log", "1"); // Pass parameters directly to cbc
    /// let result = model.with(constraint!(x + y <= 0.5)).solve();
    /// assert_eq!(result.unwrap().value(x), 0.5);
    /// ```
    pub fn set_parameter(&mut self, key: &str, value: &str) {
        self.model.set_parameter(key, value);
    }
}

impl SolverModel for CoinCbcProblem {
    type Solution = CoinCbcSolution;
    type Error = ResolutionError;

    fn solve(mut self) -> Result<Self::Solution, Self::Error> {
        // Due to a bug in cbc, SOS constraints are only taken into account
        // if the model has at least one integer variable.
        // See: https://github.com/coin-or/Cbc/issues/376
        if self.has_sos {
            // We need to add two columns to work around yet another bug
            // See: https://github.com/coin-or/Cbc/issues/376#issuecomment-803057782
            let dummy_col1 = self.model.add_col();
            let dummy_col2 = self.model.add_col();
            self.model.set_obj_coeff(dummy_col1, 1e-6);
            self.model.set_obj_coeff(dummy_col2, 1e-6);
            self.model.set_integer(dummy_col1);
            let dummy_row = self.model.add_row();
            self.model.set_weight(dummy_row, dummy_col1, 1.);
            self.model.set_weight(dummy_row, dummy_col2, 1.);
            self.model.set_row_upper(dummy_row, 1.);
        }

        if let Some(mip_gap) = self.mip_gap {
            self.set_parameter("ratiogap", &mip_gap.to_string());
        }

        let solution = self.model.solve();
        let raw = solution.raw();
        match raw.status() {
            Status::Stopped => {
                match raw.secondary_status() {
                    SecondaryStatus::StoppedOnTime => {
                        let solution_vec = solution.raw().col_solution().into();
                        Ok(CoinCbcSolution{ status: SolutionStatus::TimeLimit, solution, solution_vec })
                    },
                    _ => Err(ResolutionError::Other("Stopped"))
                }
            },
            Status::Abandoned => Err(ResolutionError::Other("Abandoned")),
            Status::UserEvent => Err(ResolutionError::Other("UserEvent")),
            Status::Finished // The optimization finished, but may not have found a solution
            | Status::Unlaunched // The solver didn't have to be launched, presolve handled it
            => {
                if raw.is_continuous_unbounded() {
                    Err(ResolutionError::Unbounded)
                } else if raw.is_proven_infeasible() {
                    Err(ResolutionError::Infeasible)
                } else {
                    let solution_vec = solution.raw().col_solution().into();
                    match raw.secondary_status() {
                        SecondaryStatus::StoppedOnTime => Ok(CoinCbcSolution{ status: SolutionStatus::TimeLimit, solution, solution_vec }) ,
                        SecondaryStatus::StoppedOnGap => Ok(CoinCbcSolution{ status: SolutionStatus::GapLimit, solution, solution_vec }) ,
                        _ => Ok(CoinCbcSolution { status: SolutionStatus::Optimal, solution, solution_vec, }),
                    }
                }
            },
        }
    }

    fn add_constraint(&mut self, constraint: Constraint) -> ConstraintReference {
        let index = self.model.num_rows().try_into().unwrap();
        let row = self.model.add_row();
        let constant = -constraint.expression.constant;
        if constraint.is_equality {
            self.model.set_row_equal(row, constant);
        } else {
            self.model.set_row_upper(row, constant);
        }
        for (var, coeff) in constraint.expression.linear.coefficients.into_iter() {
            self.model.set_weight(row, self.columns[var.index()], coeff);
        }
        ConstraintReference { index }
    }

    fn name() -> &'static str {
        "Coin Cbc"
    }
}

impl WithInitialSolution for CoinCbcProblem {
    fn with_initial_solution(
        mut self,
        solution: impl IntoIterator<Item = (Variable, f64)>,
    ) -> Self {
        for (var, val) in solution {
            self.model
                .set_col_initial_solution(self.columns[var.index()], val);
        }
        self
    }
}

impl WithTimeLimit for CoinCbcProblem {
    fn with_time_limit<T: Into<f64>>(mut self, seconds: T) -> Self {
        self.model
            .set_parameter("sec", &(seconds.into().ceil() as usize).to_string());
        self
    }
}

/// Unfortunately, the current version of cbc silently ignores
/// sos constraints on continuous variables.
/// See <https://github.com/coin-or/Cbc/issues/376>
impl ModelWithSOS1 for CoinCbcProblem {
    fn add_sos1<I: IntoAffineExpression>(&mut self, variables: I) {
        let columns = std::mem::take(&mut self.columns);
        let cols_and_weights = variables
            .linear_coefficients()
            .into_iter()
            .map(|(var, weight)| (columns[var.index()], weight));
        self.model.add_sos1(cols_and_weights);
        self.columns = columns;
        self.has_sos = true;
    }
}

/// A coin-cbc problem solution
pub struct CoinCbcSolution {
    status: SolutionStatus,
    solution: CbcSolution,
    solution_vec: Vec<f64>, // See: rust-or/good_lp#6
}

impl CoinCbcSolution {
    /// Returns the inner Coin-Cbc model
    pub fn model(&self) -> &coin_cbc::raw::Model {
        self.solution.raw()
    }
}

impl Solution for CoinCbcSolution {
    fn status(&self) -> SolutionStatus {
        self.status
    }
    fn value(&self, variable: Variable) -> f64 {
        // Our indices should always match those of cbc
        self.solution_vec[variable.index()]
    }
}

impl WithMipGap for CoinCbcProblem {
    fn mip_gap(&self) -> Option<f32> {
        self.mip_gap
    }

    fn with_mip_gap(mut self, mip_gap: f32) -> Result<Self, MipGapError> {
        if mip_gap.is_sign_negative() {
            Err(MipGapError::Negative)
        } else if mip_gap.is_infinite() {
            Err(MipGapError::Infinite)
        } else {
            self.mip_gap = Some(mip_gap);
            Ok(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::coin_cbc;
    use crate::{
        constraint,
        solvers::{SolutionStatus, WithTimeLimit},
        variable, variables, Expression, Solution, SolverModel, Variable, WithInitialSolution,
        WithMipGap,
    };
    use float_eq::assert_float_eq;

    #[test]
    fn solve_problem_with_time_limit() {
        eprintln!("Testing time limit...");
        let n = 10;
        let mut vars = variables!();
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(vars.add(variable().binary()));
        }
        let pb = vars
            .maximise(v.iter().map(|&v| 3.5 * v).sum::<Expression>())
            .using(coin_cbc)
            .with_time_limit(0.0);
        let sol = pb.solve().unwrap();
        assert!(matches!(sol.status(), SolutionStatus::TimeLimit));
        for var in v {
            assert_eq!(sol.value(var), 1.0);
        }
    }

    #[test]
    fn can_solve_with_gap_limit() {
        let (status_optimal, value_optimal) = sat_value(None);
        let (status_suboptimal, value_suboptimal) = sat_value(Some(0.5));

        assert!(matches!(status_optimal, SolutionStatus::Optimal));
        assert!(matches!(status_suboptimal, SolutionStatus::GapLimit));
        assert!(value_suboptimal < value_optimal);
    }

    fn sat_value(mipgap: Option<f32>) -> (SolutionStatus, f64) {
        let mut prob_vars = variables!();
        let mut vars: Vec<Variable> = Vec::with_capacity(VARIABLE_COUNT);

        for _ in 0..VARIABLE_COUNT {
            let var = prob_vars.add(variable().binary());
            vars.push(var);
        }

        let objective: Expression = vars.iter().sum();
        let mut model = CLAUSES.iter().copied().fold(
            prob_vars.maximise(objective.clone()).using(coin_cbc),
            |model, (x0, x1, x2)| {
                let c0 = vars[(x0.abs() - 1) as usize];
                let c1 = vars[(x1.abs() - 1) as usize];
                let c2 = vars[(x2.abs() - 1) as usize];
                let x0 = if x0 < 0 { 1 - c0 } else { c0.into() };
                let x1 = if x1 < 0 { 1 - c1 } else { c1.into() };
                let x2 = if x2 < 0 { 1 - c2 } else { c2.into() };
                model.with(constraint!(x0 + x1 + x2 >= 1))
            },
        );

        if let Some(gap) = mipgap {
            model = model.with_mip_gap(gap).unwrap();
        }

        let solution = model.solve().unwrap();
        (solution.status(), objective.eval_with(&solution))
    }

    #[test]
    fn solve_problem_with_initial_solution() {
        let limit = 3.0;
        // Solve problem once
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
        // Recreate problem and solve with initial solution
        let initial_solution = vec![(v, sol.value(v))];
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars
            .maximise(v)
            .using(coin_cbc)
            .with_initial_solution(initial_solution);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
    }

    #[test]
    fn solve_problem_with_initial_variable_values() {
        let limit = 3.0;
        // Solve problem once
        variables! {
            vars:
                0.0 <= v <= limit;
        };
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
        // Recreate problem and solve with initial solution
        let mut vars = variables!();
        let v = vars.add(variable().min(0).max(limit).initial(2));
        let pb = vars.maximise(v).using(coin_cbc);
        let sol = pb.solve().unwrap();
        assert_float_eq!(sol.value(v), limit, abs <= 1e-8);
    }

    const VARIABLE_COUNT: usize = 75;
    const CLAUSES: &[(i32, i32, i32)] = &[
        (42, 22, 15),
        (73, -22, -24),
        (50, -20, 53),
        (-66, 49, -15),
        (39, 8, -74),
        (66, 72, -61),
        (-32, -13, -8),
        (44, -22, -1),
        (-42, 35, -20),
        (-40, -42, 57),
        (-3, -52, 7),
        (-50, -14, 34),
        (-39, -37, -28),
        (10, 25, 31),
        (-3, 45, -75),
        (69, -10, 39),
        (-22, 47, 34),
        (62, 67, -64),
        (-11, -39, 72),
        (-75, -70, -47),
        (61, -47, -65),
        (43, -70, -33),
        (28, -37, 53),
        (-63, 62, -46),
        (62, 28, 33),
        (13, 12, 57),
        (64, 4, -23),
        (-7, -32, 34),
        (66, 36, 71),
        (74, -7, 51),
        (41, -53, -15),
        (-50, -44, -53),
        (70, -72, -14),
        (37, 47, -23),
        (56, -7, 50),
        (-34, 4, -67),
        (23, -1, 15),
        (-58, 67, -2),
        (58, 56, 29),
        (49, 69, 43),
        (36, 59, 10),
        (74, -55, 35),
        (19, 38, -24),
        (2, 17, 24),
        (66, 70, -30),
        (39, -53, 48),
        (4, 51, -42),
        (-36, 68, -22),
        (57, -43, -13),
        (1, 59, -4),
        (-73, -53, 22),
        (53, 62, 68),
        (-18, 40, -16),
        (-29, 2, -10),
        (15, -13, -57),
        (1, 45, 34),
        (40, 68, 4),
        (-55, -29, -47),
        (-37, -21, 68),
        (44, 2, -17),
        (-7, -20, -33),
        (-57, -7, -40),
        (-67, 18, -4),
        (31, 50, 17),
        (-28, 65, 5),
        (-34, -21, 43),
        (39, 43, -16),
        (-54, -19, 72),
        (-54, 13, 59),
        (74, 64, 69),
        (-28, 66, 71),
        (20, 56, -21),
        (-2, 26, 71),
        (37, -21, -64),
        (31, 44, -62),
        (-62, -67, 22),
        (62, 68, 15),
        (6, -17, -48),
        (-6, -37, -1),
        (-63, -24, 26),
        (-43, 46, -19),
        (-25, 14, 60),
        (62, 26, 17),
        (15, 33, -53),
        (73, -15, -70),
        (-50, 31, -24),
        (-61, 45, 64),
        (-45, 70, -33),
        (-4, -44, -17),
        (27, 56, -57),
        (75, -53, 1),
        (-73, 32, -66),
        (69, 46, 70),
        (11, 36, -66),
        (-70, -44, 31),
        (21, 65, -12),
        (43, -11, 14),
        (34, 65, 61),
        (-50, -27, -54),
        (-16, 48, 39),
        (4, -45, 17),
        (74, -35, 39),
        (27, -26, -25),
        (31, -28, -69),
        (-4, -9, -20),
        (70, -58, 13),
        (41, 72, 39),
        (-39, -35, 57),
        (31, 3, -63),
        (71, -16, -65),
        (-59, -53, 27),
        (-75, 29, 22),
        (-69, -72, 65),
        (-11, -16, -39),
        (-55, -34, -66),
        (-13, 32, 26),
        (65, -28, -43),
        (40, 59, -71),
        (66, -45, 72),
        (-5, -22, -69),
        (-44, -3, 74),
        (-9, -32, -53),
        (-50, 7, -26),
        (27, -26, -30),
        (-58, -60, 18),
        (18, 25, -11),
        (42, 27, -10),
        (62, 52, 58),
        (73, -50, 44),
        (-23, 55, -9),
        (-49, -48, 13),
        (-37, -30, -58),
        (23, -35, -37),
        (-23, -45, 34),
        (16, -19, -10),
        (13, 33, -12),
        (-56, 53, 29),
        (-23, -47, 37),
        (44, -74, -54),
        (-33, 6, -68),
        (7, -42, 10),
        (-18, 47, 4),
        (49, 32, -74),
        (-20, -60, -49),
        (-48, -20, -44),
        (18, 69, 40),
        (-52, 30, 1),
        (8, -50, -42),
        (55, 5, 43),
        (67, 50, 36),
        (-33, -31, 73),
        (37, -71, -29),
        (24, 46, -5),
        (9, 34, -31),
        (-28, 12, -36),
        (22, -60, -28),
        (-46, 5, -65),
        (-38, -45, -31),
        (-30, -7, -20),
        (47, -72, 67),
        (4, 6, -3),
        (25, -71, -52),
        (75, -32, -42),
        (16, 28, 47),
        (16, -60, -12),
        (26, 70, 2),
        (-75, -50, 21),
        (48, 64, -46),
        (54, -35, 72),
        (-41, 10, -69),
        (-48, -43, 54),
        (74, -70, -54),
        (34, 18, -32),
        (-48, 11, 10),
        (-54, -2, 14),
        (32, -25, -73),
        (-15, 72, 63),
        (-37, -49, 51),
        (-43, -25, -17),
        (-55, 47, 54),
        (23, -50, -56),
        (14, -1, -50),
        (34, 30, 38),
        (15, -65, 40),
        (-9, 20, 4),
        (-19, -35, -72),
        (53, -68, -58),
        (-18, 27, 6),
        (71, 51, -18),
        (-67, 16, 3),
        (17, 21, 50),
        (-71, 73, 22),
        (28, 42, -34),
        (-18, -52, -4),
        (43, 75, 64),
        (-54, -24, 52),
        (-62, -40, -27),
        (67, -32, 8),
        (12, -71, -61),
        (-1, -15, -44),
        (-67, 30, 69),
        (50, -7, -2),
        (-42, -36, -22),
        (-59, 72, -71),
        (-14, 12, 57),
        (28, 68, 62),
        (-59, 11, -14),
        (-51, 54, 19),
        (-72, -12, 2),
        (34, -69, 25),
        (48, 4, -19),
        (-41, -9, 66),
        (-37, -45, 57),
        (-61, 38, 21),
        (75, 64, -69),
        (-50, 51, -48),
        (-48, 51, -6),
        (41, -66, 44),
        (32, 48, -52),
        (-17, -67, 47),
        (16, -46, -63),
        (56, -31, -66),
        (60, 50, -33),
        (39, -53, 15),
        (-68, -12, 8),
        (-54, 68, 49),
        (26, -56, -60),
        (-6, -59, -19),
        (6, -29, 17),
        (-37, 23, -45),
        (54, -68, -22),
        (-46, -49, 35),
        (67, 32, -23),
        (-36, 52, 3),
        (-18, -57, 38),
        (-35, 30, 57),
        (25, -71, -67),
        (3, 1, 17),
        (-69, 74, 6),
        (-67, -75, 58),
        (-14, -15, -50),
        (26, 18, 68),
        (12, -31, -37),
        (31, 57, 58),
        (-58, -22, -1),
        (60, 14, -71),
        (31, -34, 7),
        (60, 63, -66),
        (-6, 47, -36),
        (66, -12, 30),
        (25, 43, 48),
        (-67, -39, 27),
        (-30, -65, -34),
        (17, -4, 28),
        (-16, 5, -9),
        (59, -41, -24),
        (-35, 64, 19),
        (72, 61, -7),
        (70, -47, -54),
        (-40, -12, -52),
        (-30, -41, 34),
        (-34, 31, -21),
        (69, -13, -37),
        (50, -37, 39),
        (-1, -24, 47),
        (46, 27, -69),
        (-35, 38, 21),
        (-32, 8, 49),
        (60, 58, -54),
        (59, -12, -62),
        (-60, 34, 11),
        (-43, -72, -54),
        (-24, 8, 1),
        (-15, 61, 67),
        (31, 68, 14),
        (1, -8, -35),
        (27, 28, -69),
        (-72, 39, -53),
        (-6, -51, -39),
        (74, -40, 12),
        (17, -71, -48),
        (-54, 8, 38),
        (55, -61, -52),
        (-72, -3, -55),
        (-39, 15, 65),
        (70, -74, 51),
        (-1, 62, -57),
        (38, 70, 50),
        (-73, -13, -15),
        (30, -16, 4),
        (-23, 64, -5),
        (-17, -11, -46),
        (63, -31, -50),
        (-50, 42, -4),
        (-54, -48, -70),
        (-63, 15, -53),
        (52, 30, -74),
        (-3, -16, -22),
        (-43, 26, -34),
        (26, -68, 25),
        (-9, 38, -32),
        (-49, 7, 73),
        (-38, 31, 41),
        (52, 50, -44),
        (62, -46, 51),
        (-29, -75, 63),
        (29, 34, -25),
        (52, 43, 9),
        (35, 4, 66),
        (61, 75, -57),
        (-62, 40, 46),
        (68, 54, -64),
        (69, -58, -37),
        (18, 34, 67),
        (20, -22, 5),
        (47, 45, 55),
        (48, 12, -58),
        (-31, 22, -45),
        (-55, -42, -4),
        (-51, -69, -61),
        (-70, -41, 14),
        (54, -31, -44),
        (-25, 22, -18),
        (-39, -68, -8),
        (-6, -15, -51),
    ];
}
