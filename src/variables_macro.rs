#[macro_export]
macro_rules! variables {
    () => { $crate::variable::ProblemVariables::__new_internal(||()) }
}
// The line number of the closure is inserted into error messages.
// Do not change the line number of the __new_internal call to not break the compilation tests