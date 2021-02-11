use good_lp::variables;

fn sametype<F>(_a: F, _b: F) {}

fn main() {
    let mut var1 = variables!();
    let mut var2 = variables!();
    sametype(var1, var2)
}