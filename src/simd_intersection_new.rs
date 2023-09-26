use setops::intersect::{galloping_avx512, qfilter};
use setops::visitor::{Counter, SliceWriter, VecWriter};
use std::mem;

#[inline(always)]
pub fn intersect_simd_gallop(aaa: &[u32], bbb: &[u32], results: Option<&mut Vec<u32>>) -> usize {
    if let Some(vec) = results {
        let mut writer =  VecWriter::<i32>::new();

        unsafe {

            galloping_avx512(
                mem::transmute::<&[u32], &[i32]>(aaa),
                mem::transmute::<&[u32], &[i32]>(bbb),
                &mut writer,
            );
        }

        vec.extend(writer.as_ref().iter().map(|&a| a as u32));

        writer.as_ref().len()
    } else {
        let mut counter = Counter::new();

        unsafe {
            galloping_avx512(
                mem::transmute::<&[u32], &[i32]>(aaa),
                mem::transmute::<&[u32], &[i32]>(bbb),
                &mut counter,
            );
        }

        counter.count()
    }
}

#[inline(always)]
pub fn intersect_simd_qfilter(aaa: &[u32], bbb: &[u32], results: Option<&mut Vec<u32>>) -> usize {
    // unimplemented!()
    if let Some(vec) = results {
        let mut writer = VecWriter::<i32>::new();

        unsafe {
            let mut writer =
                SliceWriter::from(mem::transmute::<&mut [u32], &mut [i32]>(vec.as_mut_slice()));
            qfilter(
                mem::transmute::<&[u32], &[i32]>(aaa),
                mem::transmute::<&[u32], &[i32]>(bbb),
                &mut writer,
            );
        }

        vec.extend(writer.as_ref().iter().map(|&a| a as u32));

        writer.as_ref().len()
    } else {
        let mut counter = Counter::new();

        unsafe {
            qfilter(
                mem::transmute::<&[u32], &[i32]>(aaa),
                mem::transmute::<&[u32], &[i32]>(bbb),
                &mut counter,
            );
        }

        counter.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd() {
        let x = vec![1, 2];
        let y = vec![1, 2, 6];
        let mut result = Vec::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec![1, 2]);
        let mut result = Vec::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec![1, 2]);

        let x = vec![1, 2, 3, 4];
        let y = vec![1, 2, 5, 6];
        let mut result = Vec::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec![1, 2]);
        let mut result = Vec::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 2);
        assert_eq!(result, vec![1, 2]);

        let x = vec![1, 2, 3, 4, 8, 9, 2_000_000_000];
        let y = vec![1, 2, 3, 4, 5, 6, 7, 2_000_000_000];
        let mut result = Vec::new();
        assert_eq!(intersect_simd_gallop(&x, &y, Some(&mut result)), 5);
        assert_eq!(result, vec![1, 2, 3, 4, 3_000_000_000]);
        let mut result = Vec::new();
        assert_eq!(intersect_simd_qfilter(&x, &y, Some(&mut result)), 5);
        assert_eq!(result, vec![1, 2, 3, 4, 3_000_000_000]);
    }
}
