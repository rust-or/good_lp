/// Instantiates [ProblemVariables](crate::variable::ProblemVariables),
/// to create a set of related variables.
///
/// Using a macro allows you to create variables with a readable syntax close to
/// the [LP file format](https://www.gurobi.com/documentation/9.1/refman/lp_format.html).
/// If you don't need that, you can instantiate your linear program with only
/// [ProblemVariables::new](crate::variable::ProblemVariables)
///
/// ## Working examples
/// ### Defining variables in the macro
///
/// The most concise way to define variables is to do it directly in the macro.
///
/// ```
/// use good_lp::{variables, default_solver, SolverModel};
///
/// variables!{problem:
///     0 <= x;  // Create a variable x that varies between 0 and +∞
///     0 <= y <= 10;  // y varies between 0 and 10
///     z; // z varies between -∞ and +∞
/// } // x, y, and z are now three rust variables that are in scope
/// # #[cfg(not(feature = "coin_cbc"))] // see: https://github.com/coin-or/Cbc/issues/367
/// problem.minimise(x + y + z).using(default_solver).solve();
/// ```
/// ### Creating a vector of variables
///
/// ```
/// use good_lp::{variable, variables, Expression};
/// variables!{vars: 0 <= x[3] <= 1; } // x will be a vector of variables
/// let objective = x[0] + x[1] - x[2];
/// ```
///
/// ### Creating integer variables
///
/// ```
/// # use good_lp::{variable, variables};
/// variables!{vars: 0 <= x[3] (integer)  <= 8; } // x will be a vector of integer variables
/// ```
///
/// ### Creating binary variables
///
/// ```
/// use good_lp::{variable, variables};
/// variables!{vars: x (binary); }
/// // equivalent to:
/// variables!{vars: 0 <= x (integer) <= 1; }
/// ```
///
/// ### Simply instantiating  [ProblemVariables](crate::variable::ProblemVariables)
/// ```
/// use good_lp::{variable, variables, Expression};
/// let mut vars = variables!();
/// let x = vars.add(variable().min(0));
/// let y = vars.add(variable().max(9));
/// let objective = x + y;
/// ```
///
/// ### Defining variables programmatically
///
/// Sometimes you don't know before run time how many variables you are going to have.
/// In these cases, you can use the methods in [ProblemVariables](crate::variable::ProblemVariables)
/// to dynamically add variables to your problem.
///
/// ```
/// use good_lp::{variable, variables, Expression};
/// # let should_add_y = true;
/// variables!{vars: 0 <= x; } // The variable x will always be present
///
/// let y = if should_add_y { // The variable y will be present only if the condition is true
///    Some(vars.add(variable().min(0)))
/// } else {None};
///
/// let objective = x + y.map(Expression::from).unwrap_or_default();
/// // objective is now x + y if should_add_y, and just x otherwise
/// ```
///
/// ### Setting bounds from outside expressions
///
/// Because of restrictions on rust macros, this works :
///
/// ```
/// # use good_lp::variables;
/// let max_x = 10;
/// variables!{vars: x <= max_x; } // max_x is the upper bound for x
/// ```
///
/// But this doesn't:
/// ```compile_fail
/// # use good_lp::variables;
/// let min_x = 10;
/// variables!{vars: min_x <= x; } // trying to set min_x as the lower bound for x, but this fails
/// ```
///
/// If you want to use a value computed outside of the macro invocation as a lower bound,
/// use this syntax:
/// ```
/// # use good_lp::variables;
/// let min_x = 10;
/// variables!{vars: x >= min_x; } // min_x is the lower bound for x
/// ```
/// ## Trying to add incompatible variables
///
/// You should never create expressions with variables that come from different
/// [ProblemVariables](crate::variable::ProblemVariables) instances.
///
/// ```should_panic,ignore-wasm
/// use good_lp::{variables, default_solver, SolverModel};
///
/// variables!{pb1: a;}
/// variables!{pb2: x; y;} // Creating my variables on pb2 ...
/// pb1.minimise(x + y) // ... but running the optimization on pb1
///   .using(default_solver)
///   .solve();
/// ```
/// Since `pb1` and `pb2` are different problems, their variables are not compatible with one another.
/// Trying to solve problems with incompatible variables will **panic**.
#[macro_export]
macro_rules! variables {
    () => {$crate::variable::ProblemVariables::new()};
    (
    $vars:ident:
    $(
        $($min:literal <= )?
        $var_name:ident
        $([$length:expr])?
        $(($qualifier:tt))?
        $(<= $max:expr;)?
        $(>= $postfix_min:expr;)?
        $(;)?
    )*
    ) => {
            let mut $vars = $crate::variable::ProblemVariables::new();
            $(
                let $var_name = {
                    let var_def = $crate::variable()
                                .name(stringify!($var_name))
                                $(.min($min))*
                                $(.max($max))*
                                $(.min($postfix_min))*
                                $(.$qualifier())*;
                    $crate::variables!(@add_variable, $vars, var_def, $($length)*)
                };
            )*
        };
    (@add_variable, $vars:expr, $var:expr, $length:expr) => {
        $vars.add_vector($var, $length)
    };
    (@add_variable, $vars:expr, $var:expr,) => {
        $vars.add($var)
    };
}
