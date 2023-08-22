#[allow(unused_macros)]
macro_rules! mipgap_tests {
    ($solver:expr) => {
        #[test]
        fn mipgap_default() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver);
            assert_eq!(model.mip_gap(), None);
        }

        #[test]
        fn mipgap_pos_zero() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(0.0).unwrap();
            assert_eq!(model.mip_gap(), Some(0.0));
        }

        #[test]
        fn mipgap_neg_zero() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(-0.0);
            assert!(model.is_err());
        }

        #[test]
        fn mipgap_pos_nonzero() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(0.5).unwrap();
            assert_eq!(model.mip_gap(), Some(0.5));
        }

        #[test]
        fn mipgap_neg_nonzero() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(-0.5);
            assert!(model.is_err());
        }

        #[test]
        fn mipgap_pos_infinity() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(f32::INFINITY);
            assert!(model.is_err());
        }

        #[test]
        fn mipgap_neg_infinity() {
            $crate::variables! { vars: a <= 1; };
            let model = vars
                .maximise(a)
                .using($solver)
                .with_mip_gap(f32::NEG_INFINITY);
            assert!(model.is_err());
        }
    };
}
