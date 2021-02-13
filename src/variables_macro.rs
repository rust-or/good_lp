// The line number of the closure is inserted into error messages.
// Do not change the line number of the __new_internal call to not break the compilation tests

/// Instantiates [ProblemVariables](crate::variable::ProblemVariables),
/// to create a set of related variables.
///
/// Using a macro allows this crate to give different types to variables that were instantiated
/// at different places in the code.
///
/// ## Working example
///
/// ```
/// use good_lp::variables;
///
/// let mut vars = variables!();
/// let x = vars.add_variable();
/// let y = vars.add_variable();
/// let objective = x + y / 2;
/// ```
///
/// ## Trying to add incompatible variables
///
/// You should never create expressions with variabless that come from different
/// [ProblemVariables](crate::variable::ProblemVariables) insstances.
///
/// ```compile_fail
/// use good_lp::variables;
///
/// let mut pb1 = variables!();
/// let mut pb2 = variables!();
/// let x = pb1.add_variable();
/// let y = pb2.add_variable();
/// let objective = x + y / 2;
/// ```
/// Since `pb1` and `pb2` have been instanciated at two different places in the code,
/// they are different problems and their variables are not compatible with one another.
#[macro_export]
macro_rules! variables {
    () => {
        $crate::variable::ProblemVariables::__new_internal(|| ())
    };
}
