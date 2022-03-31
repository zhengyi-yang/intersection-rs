use std::borrow::Cow;
use std::cmp::Ordering;
use std::env;
use std::mem;

#[cfg(feature = "simd")]
use crate::simd_intersection::{intersect_simd_gallop, intersect_simd_qfilter};

const INTERSECTION_GALLOP_OVERHEAD: usize = 4;

lazy_static! {
    /// Default magic gallop overhead # is 4
    static ref GALLOP_OVERHEAD: usize = env::var("INTERSECTION_GALLOP_OVERHEAD").map(|n| n.parse().unwrap()).unwrap_or(INTERSECTION_GALLOP_OVERHEAD);

}

#[inline(always)]
pub fn intersect_multi(mut to_intersect: Vec<Cow<[u32]>>) -> Vec<u32> {
    if to_intersect.len() == 1 {
        return to_intersect[0].iter().copied().collect();
    }

    to_intersect.sort_unstable_by_key(|x| x.len());

    let mut intersected = Vec::with_capacity(to_intersect[0].len());
    intersect(&to_intersect[0], &to_intersect[1], Some(&mut intersected));
    let mut buffer = Vec::with_capacity(intersected.len());

    let mut count;

    for candidates in to_intersect.into_iter().skip(2) {
        count = intersect(&intersected, &candidates, Some(&mut buffer));

        if count == 0 {
            return buffer;
        }

        mem::swap(&mut intersected, &mut buffer);
        buffer.clear();
    }

    intersected
}

#[inline(always)]
pub fn intersect(aaa: &[u32], bbb: &[u32], results: Option<&mut Vec<u32>>) -> usize {
    if aaa.len() < bbb.len() / *GALLOP_OVERHEAD {
        #[cfg(feature = "simd")]
        {
            intersect_simd_gallop(aaa, bbb, results)
        }
        #[cfg(not(feature = "simd"))]
        {
            intersect_scalar_gallop(aaa, bbb, results)
        }
    } else {
        #[cfg(feature = "simd")]
        {
            intersect_simd_qfilter(aaa, bbb, results)
        }
        #[cfg(not(feature = "simd"))]
        {
            intersect_scalar_merge(aaa, bbb, results)
        }
    }
}

#[inline(always)]
pub fn intersect_scalar_merge<T: Copy + Ord>(
    aaa: &[T],
    mut bbb: &[T],
    mut results: Option<&mut Vec<T>>,
) -> usize {
    let mut count = 0;

    for &a in aaa {
        while !bbb.is_empty() && bbb[0] < a {
            bbb = &bbb[1..];
        }
        if !bbb.is_empty() && a == bbb[0] {
            count += 1;
            if let Some(vec) = results.as_mut() {
                vec.push(a);
            }
        }
    }

    count
}

#[inline(always)]
pub fn intersect_scalar_gallop<T: Copy + Ord>(
    aaa: &[T],
    mut bbb: &[T],
    mut results: Option<&mut Vec<T>>,
) -> usize {
    let mut count = 0;

    for a in aaa {
        bbb = gallop(bbb, a);
        if !bbb.is_empty() && &bbb[0] == a {
            count += 1;
            if let Some(vec) = results.as_mut() {
                vec.push(*a);
            }
        }
    }

    count
}

/// The `gallop` binary searching algorithm.
/// **Note** it is necessary to guarantee that `slice` is sorted.
///
/// # Parameters
///
/// * `slice`: The ordered slice to search in.
/// * `cmp`: A udf comparing function, which indicates `key.cmp(&slice[i])`
///
/// # Return
///
/// A sub-slice `SL` of the input slice such that `cmp(SL[0]) != Ordering::Greater`.
///
/// # Example
///
/// ```
/// extern crate intersection;
///
/// use intersection::intersect;
///
/// let vec = vec![1, 2, 3];
/// let key = 2;
///
/// let slice = intersect::gallop_by(&vec, |x: &u32| key.cmp(x));
/// assert_eq!(slice.to_vec(), vec![2, 3]);
/// ```
#[inline(always)]
pub fn gallop_by<T>(mut slice: &[T], mut cmp: impl FnMut(&T) -> Ordering) -> &[T] {
    // if empty slice, or key <= slice[0] already, return
    if !slice.is_empty() && cmp(&slice[0]) == Ordering::Greater {
        let mut step = 1;
        while step < slice.len() && cmp(&slice[step]) == Ordering::Greater {
            slice = &slice[step..];
            step <<= 1;
        }

        step >>= 1;
        while step > 0 {
            if step < slice.len() && cmp(&slice[step]) == Ordering::Greater {
                slice = &slice[step..];
            }
            step >>= 1;
        }

        slice = &slice[1..]; // advance one, as we always stayed < key
    }

    slice
}

#[inline(always)]
pub fn gallop_gt_by<T>(mut slice: &[T], mut cmp: impl FnMut(&T) -> Ordering + Copy) -> &[T] {
    slice = gallop_by(slice, cmp);

    while !slice.is_empty() && cmp(&slice[0]) == Ordering::Equal {
        slice = &slice[1..];
    }

    slice
}

/// The `gallop` algorithm with key.
#[inline(always)]
pub fn gallop<'a, T: Ord>(slice: &'a [T], key: &T) -> &'a [T] {
    gallop_by(slice, |x: &T| key.cmp(x))
}

/// The `gallop_gt` algorithm with key
#[inline(always)]
pub fn gallop_gt<'a, T: Ord>(slice: &'a [T], key: &T) -> &'a [T] {
    gallop_gt_by(slice, |x: &T| key.cmp(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_gallop() {
        let data = vec![1, 2, 3, 4];
        let key = 2_u32;
        let result = gallop_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![2, 3, 4]);
        let result = gallop_gt_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![3, 4]);

        let data = vec![1, 2, 5, 6];
        let key = 3_u32;
        let result = gallop_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![5, 6]);

        let data = vec![1, 1, 2, 2, 3, 3, 4, 4];
        let key = 2_u32;
        let result = gallop_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![2, 2, 3, 3, 4, 4]);
        let result = gallop_gt_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![3, 3, 4, 4]);

        let data = vec![1, 2, 3, 3, 8, 9, 10];
        let key = 4_u32;
        let result = gallop_by(&data, |x: &u32| key.cmp(x));
        assert_eq!(result.to_vec(), vec![8, 9, 10]);
    }

    #[test]
    fn test_intersect_multi() {
        println!("running test_intersect_multi");
        let v1 = vec![0, 1, 2, 3, 5, 10, 11, 20, 30];
        let v2 = vec![1, 3, 5, 10, 11, 70];

        let data = vec![Cow::from(&v1), Cow::from(&v2)];

        let intersection = intersect_multi(data);

        assert_eq!(intersection, vec![1, 3, 5, 10, 11]);
    }
}
