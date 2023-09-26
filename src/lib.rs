#[macro_use]
extern crate lazy_static;

pub mod intersect;
#[cfg(feature = "simd")]
pub mod simd_intersection;

#[cfg(feature = "simd_new")]
pub mod simd_intersection_new;

pub use crate::intersect::intersect_multi;
