cmake_minimum_required(VERSION 3.11)
project(tests)

include("./volesti/external/cmake-files/Eigen.cmake")
GetEigen()
if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
  add_compile_options(-D "DBG" )
else ()
  add_compile_definitions(" DBG" )
endif ()

include("./volesti/external/cmake-files/Boost.cmake")
GetBoost()

# Explicitly find and link GLPK
set(GLPK_INCLUDE_DIR "/usr/include")
set(GLPK_LIBRARY "/usr/lib/x86_64-linux-gnu/libglpk.so")

include_directories(${GLPK_INCLUDE_DIR})
link_directories("/usr/lib/x86_64-linux-gnu")
link_libraries(${GLPK_LIBRARY})

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

include_directories(BEFORE ./volesti/external)
include_directories(BEFORE ./volesti/include)

# Ensure GLPK paths are correctly included
include_directories(${GLPK_INCLUDE_DIRS})
link_directories(${GLPK_LIBRARY_DIRS})

# Create executable for benchmark_lp.cpp
add_executable(benchmark_lp ip_solver.cpp benchmark_lp.cpp)
target_link_libraries(benchmark_lp PUBLIC ${GLPK_LIBRARIES})

# Print paths for debugging
message(STATUS "GLPK Include Dirs: ${GLPK_INCLUDE_DIRS}")
message(STATUS "GLPK Library Dirs: ${GLPK_LIBRARY_DIRS}")
message(STATUS "GLPK Libraries: ${GLPK_LIBRARIES}")