extern crate mediumvec;
#[macro_use]
extern crate lazy_static;

pub mod intersect;
#[cfg(feature = "simd")]
pub mod simd_intersection;

pub use crate::intersect::intersect_multi;
