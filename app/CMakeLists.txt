cmake_minimum_required(VERSION 3.4.1)

#include头文件目录
include_directories(src/main/cpp/thirdparty/ncnn/include
                    src/main/cpp/include/DetectAndRecognize)

#source directory源文件目录
file(GLOB FACE_SRC src/main/cpp/include/DetectAndRecognize/*.h
                   src/main/cpp/src/*.cpp
                   src/main/cpp/src/DetectAndRecognize/*.cpp)


set(FACE_COMPILE_CODE ${FACE_SRC})

#添加ncnn库
add_library(libncnn STATIC IMPORTED )
set_target_properties(libncnn
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libncnn.a)

#编译为动态库
add_library(detectandrecognize SHARED ${FACE_COMPILE_CODE})

set_target_properties(detectandrecognize PROPERTIES LIBRARY_OUTPUT_DIRECTORY "src/main/jniLibs/armeabi-v7a")

#添加工程所依赖的库
find_library(  log-lib log )
target_link_libraries( detectandrecognize
                       libncnn
                       jnigraphics
                       z
                       ${log-lib})