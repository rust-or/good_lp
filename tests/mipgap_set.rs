use good_lp::{
    solvers::{MipGapError, WithMipGap},
    variables, Solver,
};

#[cfg(feature = "coin_cbc")]
use good_lp::coin_cbc;

#[cfg(feature = "highs")]
use good_lp::highs;

#[cfg(feature = "lp-solvers")]
use good_lp::{solvers::lp_solvers::GlpkSolver, LpSolver};

#[allow(dead_code)] // used only with some features
fn generic_mipgap_set<S>(solver: S)
where
    S: Solver,
    S::Model: WithMipGap,
{
    variables! { vars: 0 <= a (integer) <= 10; };

    let mut model = vars.maximise(a).using(solver);
    assert_eq!(model.mip_gap(), None);

    model = model.with_mip_gap(0.0).unwrap();
    assert_eq!(model.mip_gap(), Some(0.0));

    model = model.with_mip_gap(0.5).unwrap();
    assert_eq!(model.mip_gap(), Some(0.5));

    let model_err = model.with_mip_gap(-0.0);
    // Ok return type might not implement Debug so we map it to something simple
    assert_eq!(model_err.map(|_| "Success"), Err(MipGapError::Negative));
}

#[cfg(feature = "coin_cbc")]
#[test]
fn mipgap_set_coin_cbc() {
    generic_mipgap_set(coin_cbc);
}

#[cfg(feature = "highs")]
#[test]
fn mipgap_set_highs() {
    generic_mipgap_set(highs);
}

#[cfg(feature = "lp-solvers")]
#[test]
fn mipgap_set_lp_solvers() {
    generic_mipgap_set(LpSolver(GlpkSolver::new()));
}
