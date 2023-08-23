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
            // Ok return type might not implement Debug so we map it to something simple
            assert_eq!(
                model.map(|_| "Success"),
                Err($crate::solvers::MipGapError::Negative)
            );
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
            // Ok return type might not implement Debug so we map it to something simple
            assert_eq!(
                model.map(|_| "Success"),
                Err($crate::solvers::MipGapError::Negative)
            );
        }

        #[test]
        fn mipgap_pos_infinity() {
            $crate::variables! { vars: a <= 1; };
            let model = vars.maximise(a).using($solver).with_mip_gap(f32::INFINITY);
            // Ok return type might not implement Debug so we map it to something simple
            assert_eq!(
                model.map(|_| "Success"),
                Err($crate::solvers::MipGapError::Infinite)
            );
        }

        #[test]
        fn mipgap_neg_infinity() {
            $crate::variables! { vars: a <= 1; };
            let model = vars
                .maximise(a)
                .using($solver)
                .with_mip_gap(f32::NEG_INFINITY);
            // Ok return type might not implement Debug so we map it to something simple
            assert_eq!(
                model.map(|_| "Success"),
                Err($crate::solvers::MipGapError::Negative)
            );
        }
    };
}
