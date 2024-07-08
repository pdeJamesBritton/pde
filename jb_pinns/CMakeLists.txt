## jb_pinns version
# cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
# cmake --build . --config Release


cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pinn)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



# Provide the full path to the library, so CMake knows where to find it.
#set_target_properties(MyExternalLib PROPERTIES IMPORTED_LOCATION /usr/local/lib/python3.10/dist-packages/torch/lib/libc10.so)
#add_library(MyExternalLib SHARED IMPORTED)


add_executable(pinn pinn.cpp)
target_link_libraries(pinn "${TORCH_LIBRARIES}" ) #MyExternalLib)

target_include_directories(pinn PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET pinn PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pinn
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:pinn>)
endif (MSVC)