cmake_minimum_required(VERSION 3.10)

project(offt-benchmark)

if(MSVC)

  add_compile_options(
    /W4 
    $<$<CONFIG:Release>:/GL>
  )

  add_link_options(
    $<$<CONFIG:Release>:/LTCG>
  )

else()
  add_compile_options(-Wall -Wextra -Wpedantic)

  # Enforce C++11 to ensure we are not using features from later standards
  add_compile_options(-std=c++11)
endif()


add_library(fftw3 STATIC IMPORTED)
set_target_properties(fftw3 PROPERTIES
  IMPORTED_LOCATION "C:/Users/Christian Volmer/Documents/Projekte/Libraries/fftw3/libfftw3-3.lib"
  INTERFACE_INCLUDE_DIRECTORIES "C:/Users/Christian Volmer/Documents/Projekte/Libraries/fftw3"
)

add_library(mkl STATIC IMPORTED)
set_target_properties(mkl PROPERTIES
  IMPORTED_LOCATION "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64/mkl_rt.lib"
  INTERFACE_INCLUDE_DIRECTORIES "C:/Program Files (x86)/Intel/oneAPI/mkl/2023.1.0/include"
)

add_subdirectory(src)
add_subdirectory(../offt/offt offt)
