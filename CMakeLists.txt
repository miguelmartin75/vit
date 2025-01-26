cmake_minimum_required(VERSION 3.5)

project(
    vit
    LANGUAGES 
    C CXX CUDA
    VERSION 0.0.1
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

set(LIB_SRCS 
    vit.cpp
)
set(
    DEPS_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/
    ${CMAKE_SOURCE_DIR}/3rdparty/
    ${CMAKE_SOURCE_DIR}/3rdparty/fmt/include
    ${CMAKE_SOURCE_DIR}/3rdparty/loguru/
)
add_subdirectory(3rdparty/fmt)

# add_compile_options(-fno-rtti -Wall -Wextra -Wpedantic)
add_compile_options(-fno-rtti)

# TODO: if debug
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# TODO: library
# add_library(libvit SHARED ${LIB_SRCS})
add_executable(vit ${LIB_SRCS})
target_include_directories(vit PUBLIC ${DEPS_INCLUDE_DIRS})
target_link_libraries(vit PUBLIC fmt::fmt)

# TODO: tests
# find_package(Catch2 3 REQUIRED)
# add_executable(test_basetypes tests/test_basetypes.cpp)
# target_link_libraries(test_parser PRIVATE Arriba)

