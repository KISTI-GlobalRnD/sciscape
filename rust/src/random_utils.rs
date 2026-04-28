use rand::distributions::{Distribution, Uniform};
use rand::Rng;

#[inline]
pub(crate) fn fill_identity_u32(out: &mut Vec<u32>, n: usize) {
    debug_assert!(n <= u32::MAX as usize);
    if out.len() != n {
        out.resize(n, 0);
    }
    for (idx, value) in out.iter_mut().enumerate() {
        *value = idx as u32;
    }
}

/// Match CWTS Java's `Arrays.permuteRandomly` shuffle pattern.
///
/// This is intentionally not Fisher-Yates: CWTS swaps each position with a
/// uniformly sampled position from the full array. The exact RNG differs, but
/// keeping the same permutation policy makes Leiden's search path closer to
/// the Java implementation at no meaningful extra cost.
#[inline]
pub(crate) fn permute_cwts_style(order: &mut [u32], rng: &mut impl Rng) {
    let n = order.len();
    if n <= 1 {
        return;
    }

    let uniform = Uniform::new(0, n);
    for i in 0..n {
        let j = uniform.sample(rng);
        order.swap(i, j);
    }
}
