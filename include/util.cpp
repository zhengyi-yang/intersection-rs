#include "intersection/include/util.hpp"

void align_malloc(void **memptr, size_t alignment, size_t size) {
  int malloc_flag = posix_memalign(memptr, alignment, size);
  if (malloc_flag) {
    std::cerr << "posix_memalign: " << strerror(malloc_flag) << std::endl;
    quit();
  }
}