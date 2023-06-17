project(offt-speed-compare)

add_executable(speedcompare
    main.cpp
)

target_link_libraries(speedcompare PRIVATE fftw3)
target_link_libraries(speedcompare PRIVATE mkl)
target_link_libraries(speedcompare PRIVATE offt)
