## jb_pinns version
# do in build directory
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
# do in jb_pinns directory
# cmake --build . --config Release

# for debugging
# make debug_build
# cd to debug_build
# call:
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DCMAKE_BUILD_TYPE=Debug ..
#changed pinn to pinndebug
# move up a directory
# called:
# make 
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pinndebug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)
find_package(PythonLibs 3.10)
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
#add_executable(foo foo.cc)
#target_link_libraries(foo ${ARMADILLO_LIBRARIES})
add_executable(pinndebug pinn.cpp)
target_link_libraries(pinndebug "${TORCH_LIBRARIES}" ${ARMADILLO_LIBRARIES} ${PYTHON_LIBRARIES} ) #MyExternalLib)
# set(CMAKE_PREFIX_PATH "<libtorch-path>/share/cmake/Torch")





# Provide the full path to the library, so CMake knows where to find it.
#set_target_properties(MyExternalLib PROPERTIES IMPORTED_LOCATION /usr/local/lib/python3.10/dist-packages/torch/lib/libc10.so)
#add_library(MyExternalLib SHARED IMPORTED)





target_include_directories(pinndebug PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET pinndebug PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
#
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
# add_custom_command(TARGET pinndebug
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:pinn>)
##endif (MSVC)