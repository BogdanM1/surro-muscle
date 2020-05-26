cmake_minimum_required(VERSION 3.0)
project(surro_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE SRC_FILES "src/*.cpp")	

include_directories(/home/bogdan/tensorflow/include)
link_directories(/home/bogdan/tensorflow/lib)

add_executable(surro_test ${SRC_FILES})
target_link_libraries(surro_test tensorflow)


