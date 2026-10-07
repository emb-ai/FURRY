cmake_minimum_required(VERSION 3.22)
project(g1_quest LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(G1_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(MUJOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_SIMULATE OFF CACHE BOOL "" FORCE)
set(MUJOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MUJOCO_TEST_PYTHON_UTIL OFF CACHE BOOL "" FORCE)
set(MUJOCO_WITH_USD OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
add_subdirectory("${G1_ROOT}/vendor/mujoco" mujoco)
target_compile_definitions(mujoco PRIVATE _POSIX_C_SOURCE=200809L)
set_target_properties(mujoco PROPERTIES VERSION "" SOVERSION "")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${G1_ROOT}/vendor/onnxruntime-android/jni/arm64-v8a/libonnxruntime.so")
add_library(openxr_loader SHARED IMPORTED)
set_target_properties(openxr_loader PROPERTIES IMPORTED_LOCATION "${G1_ROOT}/vendor/openxr-android/jni/arm64-v8a/libopenxr_loader.so")
add_library(g1_quest SHARED
    generated/xr/main.cpp generated/xr/openxr_program.cpp generated/xr/logger.cpp
    generated/xr/platformplugin_factory.cpp generated/xr/platformplugin_android.cpp
    generated/xr/graphicsplugin_factory.cpp
    native/graphics.cpp native/simulation.cpp native/gmr.cpp native/meta_retarget.cpp native/quest_runtime.cpp
    "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c")
target_compile_definitions(g1_quest PRIVATE XR_USE_PLATFORM_ANDROID XR_USE_GRAPHICS_API_OPENGL_ES DEFAULT_GRAPHICS_PLUGIN_OPENGLES)
target_include_directories(g1_quest PRIVATE
    generated/xr native "${ANDROID_NDK}/sources/android/native_app_glue"
    "${G1_ROOT}/vendor/openxr-android/prefab/modules/headers/include"
    "${G1_ROOT}/vendor/onnxruntime-android/headers"
    "${G1_ROOT}/vendor/OpenXR-SDK-Source/src"
    "${G1_ROOT}/vendor/OpenXR-SDK-Source/src/common")
target_link_libraries(g1_quest PRIVATE mujoco onnxruntime openxr_loader android log EGL GLESv3)
target_link_options(g1_quest PRIVATE "-uANativeActivity_onCreate" "-Wl,-z,max-page-size=16384")

add_executable(g1_check native/check.cpp native/simulation.cpp)
target_include_directories(g1_check PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_check PRIVATE mujoco onnxruntime log)
add_executable(g1_tracking_check native/check_tracking.cpp native/simulation.cpp native/retarget.cpp)
target_include_directories(g1_tracking_check PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_tracking_check PRIVATE mujoco onnxruntime log)
add_executable(g1_tracking_stress native/diagnose_tracking.cpp native/simulation.cpp native/retarget.cpp)
target_include_directories(g1_tracking_stress PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_tracking_stress PRIVATE mujoco onnxruntime log)
add_executable(g1_recording_check native/check_recording.cpp native/simulation.cpp native/retarget.cpp)
target_include_directories(g1_recording_check PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_recording_check PRIVATE mujoco onnxruntime log)
add_executable(g1_replay_human native/replay_human.cpp native/retarget.cpp)
target_link_libraries(g1_replay_human PRIVATE mujoco)
add_executable(g1_tracking_frames_check native/check_tracking_frames.cpp native/simulation.cpp native/retarget.cpp)
target_include_directories(g1_tracking_frames_check PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_tracking_frames_check PRIVATE mujoco onnxruntime log)
add_executable(g1_tracking_audit native/audit_tracking.cpp native/simulation.cpp native/retarget.cpp)
target_include_directories(g1_tracking_audit PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_tracking_audit PRIVATE mujoco onnxruntime log)

add_executable(g1_gmr_oracle native/gmr_oracle.cpp native/gmr.cpp)
target_link_libraries(g1_gmr_oracle PRIVATE mujoco)
add_executable(g1_meta_check native/check_meta_retarget.cpp native/gmr.cpp native/meta_retarget.cpp native/simulation.cpp)
target_include_directories(g1_meta_check PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_meta_check PRIVATE mujoco onnxruntime log)

add_executable(g1_replay_gmr native/replay_gmr.cpp native/gmr.cpp native/meta_retarget.cpp native/simulation.cpp)
target_include_directories(g1_replay_gmr PRIVATE "${G1_ROOT}/vendor/onnxruntime-android/headers")
target_link_libraries(g1_replay_gmr PRIVATE mujoco onnxruntime log)
