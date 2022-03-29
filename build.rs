fn main() {
    #[cfg(feature = "simd")]
    {
        build_cxx();
    }
}

#[cfg(feature = "simd")]
fn build_cxx() {
    cxx_build::bridge("src/simd_intersection.rs")
        .file("include/intersection_algos.cpp")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-mavx2")
        .flag_if_supported("-lmetis")
        .opt_level(3)
        .compile("simd_intersection");

    println!("cargo:rerun-if-changed=src/simd_intersection.rs");
    println!("cargo:rerun-if-changed=include/intersection_algos.cpp");
    println!("cargo:rerun-if-changed=include/intersection_algos.hpp");
    println!("cargo:rerun-if-changed=include/util.cpp");
    println!("cargo:rerun-if-changed=include/util.hpp");
}
