use good_lp::variables;

fn main() {
    let mut pb1 = variables!();
    let mut pb2 = variables!();
    let a = pb1.add_variable();
    let b = pb2.add_variable();
    let _result = a + b;
}