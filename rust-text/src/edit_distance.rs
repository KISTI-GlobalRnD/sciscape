//! Levenshtein edit distance — Wagner-Fischer with early exit.

/// Levenshtein distance with early-exit when difference exceeds threshold.
#[inline]
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();

    if a_len == 0 { return b_len; }
    if b_len == 0 { return a_len; }

    // Ensure a is the longer string for single-row optimization
    let (a, b, a_len, b_len) = if a_len < b_len {
        (b, a, b_len, a_len)
    } else {
        (a, b, a_len, b_len)
    };

    let b_chars: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0usize; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1)
                .min(prev[j + 1] + 1)
                .min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Edit distance with max threshold (early exit).
#[inline]
pub fn edit_distance_threshold(a: &str, b: &str, max_dist: usize) -> Option<usize> {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    if a_len.abs_diff(b_len) > max_dist {
        return None;
    }
    let d = edit_distance(a, b);
    if d <= max_dist { Some(d) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_threshold() {
        assert_eq!(edit_distance_threshold("cat", "car", 1), Some(1));
        assert_eq!(edit_distance_threshold("cat", "dog", 1), None);
    }
}
