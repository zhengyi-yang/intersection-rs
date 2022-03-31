/// https://github.com/pkumod/GraphSetIntersection/blob/master/src/intersection_algos.cpp
/// Han S, Zou L, Yu J X. Speeding up set intersections in graph algorithms using simd instructions[C]
/// Proceedings of the 2018 International Conference on Management of Data. 2018: 1587-1602.
use mediumvec::Vec32;

use crate::intersect::{intersect_scalar_gallop, intersect_scalar_merge};

#[cxx::bridge]
mod ffi {
    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("intersection/include/intersection_algos.hpp");

        unsafe fn intersect_simdgalloping_uint(
            set_a: *const u32,
            size_a: i32,
            set_b: *const u32,
            size_b: i32,
            set_c: *mut u32,
            count_only: bool,
        ) -> i32;

        unsafe fn intersect_qfilter_uint_b4(
            set_a: *const u32,
            size_a: i32,
            set_b: *const u32,
            size_b: i32,
            set_c: *mut u32,
            count_only: bool,
        ) -> i32;

        // unsafe fn intersect_shuffle_uint_b4(
        //     set_a: *const i32,
        //     size_a: i32,
        //     set_b: *const i32,
        //     size_b: i32,
        //     set_c: *mut i32,
        // ) -> i32;

        // unsafe fn intersect_qfilter_uint_b4_v2(
        //     set_a: *const i32,
        //     size_a: i32,
        //     set_b: *const i32,
        //     size_b: i32,
        //     set_c: *mut i32,
        // ) -> i32;
    }
}

#[inline(always)]
pub fn intersect_simd_gallop(aaa: &[u32], bbb: &[u32], results: Option<&mut Vec<u32>>) -> usize {
    if aaa.len() < 4 {
        return intersect_scalar_gallop(aaa, bbb, results);
    }

    if let Some(vec) = results {
        vec.reserve_exact(aaa.len());

        let count = unsafe {
            ffi::intersect_simdgalloping_uint(
                aaa.as_ptr(),
                aaa.len() as i32,
                bbb.as_ptr(),
                bbb.len() as i32,
                vec.as_mut_ptr(),
                false,
            ) as usize
        };

        unsafe {
            vec.set_len(count);
        }

        count
    } else {
        unsafe {
            ffi::intersect_simdgalloping_uint(
                aaa.as_ptr(),
                aaa.len() as i32,
                bbb.as_ptr(),
                bbb.len() as i32,
                Vec::new().as_mut_ptr(),
                true,
            ) as usize
        }
    }
}

#[inline(always)]
pub fn intersect_simd_qfilter(aaa: &[u32], bbb: &[u32], results: Option<&mut Vec32<u32>>) -> usize {
    if aaa.len() < 4 {
        return intersect_scalar_merge(aaa, bbb, results);
    }

    if let Some(vec) = results {
        vec.reserve_exact(aaa.len() as u32);

        let count = unsafe {
            ffi::intersect_qfilter_uint_b4(
                aaa.as_ptr(),
                aaa.len() as i32,
                bbb.as_ptr(),
                bbb.len() as i32,
                vec.as_mut_ptr(),
                false,
            ) as usize
        };

        unsafe {
            vec.set_len(count);
        }

        count
    } else {
        // let mut buf = Vec::with_capacity(aaa.len());
        unsafe {
            ffi::intersect_qfilter_uint_b4(
                aaa.as_ptr(),
                aaa.len() as i32,
                bbb.as_ptr(),
                bbb.len() as i32,
                Vec::new().as_mut_ptr(),
                true,
            ) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mediumvec::vec32;

    #[test]
    fn test_simd() {
        let x = vec![1, 2];
        let y = vec![1, 2, 6];
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec32![1, 2]);
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec32![1, 2]);

        let x = vec![1, 2, 3, 4];
        let y = vec![1, 2, 5, 6];
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec32![1, 2]);
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec32![1, 2]);

        let x = vec![1, 2, 3, 4, 8, 9, 3_000_000_000];
        let y = vec![1, 2, 3, 4, 5, 6, 7, 3_000_000_000];
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 5);
        assert_eq!(result, vec32![1, 2, 3, 4, 3_000_000_000]);
        let mut result = Vec32::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 5);
        assert_eq!(result, vec32![1, 2, 3, 4, 3_000_000_000]);
    }
}
