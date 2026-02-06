# 这个脚本会：
#  - 遍历 GLSL_DIR 目录下所有 *.glsl 文件（递归）
#  - 为每个 glsl 生成一个头文件到 OUTPUT_DIR 下
#  - 头文件内容形如：
#      inline constexpr const char g_shader_xxx_glsl[] = R"SHD( ... )SHD";

if(NOT DEFINED GLSL_DIR)
    message(FATAL_ERROR "GLSL_DIR not set")
endif()

if(NOT DEFINED OUTPUT_DIR)
    message(FATAL_ERROR "OUTPUT_DIR not set")
endif()

# 可选：用来作为 custom_command 的唯一 OUTPUT
# 如果传了 STAMP_FILE，就在最后写一个 stamp 文件
if(DEFINED STAMP_FILE)
    message(STATUS "embed_glsl.cmake: STAMP_FILE = ${STAMP_FILE}")
endif()

message(STATUS "embed_glsl.cmake: GLSL_DIR = ${GLSL_DIR}")
message(STATUS "embed_glsl.cmake: OUTPUT_DIR = ${OUTPUT_DIR}")

# 确保输出目录存在
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

# 递归查找所有 .glsl 文件
file(GLOB_RECURSE GLSL_FILES
        RELATIVE "${GLSL_DIR}"
        "${GLSL_DIR}/*.glsl"
)

if(GLSL_FILES STREQUAL "")
    message(WARNING "No .glsl files found in ${GLSL_DIR}")
endif()

foreach(REL_PATH IN LISTS GLSL_FILES)
    # REL_PATH 是相对路径，比如 "subdir/vertex_0.glsl" 或 "vertex_0.glsl"
    set(IN_FILE "${GLSL_DIR}/${REL_PATH}")

    # 读文件内容
    file(READ "${IN_FILE}" GLSL_CONTENT)

    # 生成变量名：g_shader_ + 相对路径（非字母数字替换为下划线）
    set(VAR_NAME "${REL_PATH}")
    string(REGEX REPLACE "[^A-Za-z0-9_]" "_" VAR_NAME "${VAR_NAME}")
    set(VAR_NAME "g_shader_${VAR_NAME}")

    # 输出头文件路径：在 OUTPUT_DIR 下保持相对目录结构，再加 .hpp
    # 比如：
    #   REL_PATH = "vertex_0.glsl" -> OUTPUT_DIR/vertex_0.glsl.hpp
    #   REL_PATH = "sub/vertex_1.glsl" -> OUTPUT_DIR/sub/vertex_1.glsl.hpp
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    if(REL_DIR STREQUAL "")
        set(OUT_DIR "${OUTPUT_DIR}")
    else()
        set(OUT_DIR "${OUTPUT_DIR}/${REL_DIR}")
    endif()

    file(MAKE_DIRECTORY "${OUT_DIR}")

    get_filename_component(FILE_NAME "${REL_PATH}" NAME)  # e.g. vertex_0.glsl
    set(OUT_FILE "${OUT_DIR}/${FILE_NAME}.hpp")

    message(STATUS "Embedding ${IN_FILE} -> ${OUT_FILE}  (var = ${VAR_NAME})")

    # 生成头文件内容，使用 raw string，避免转义 "
    set(HEADER_TEXT
            "#pragma once

inline constexpr const char ${VAR_NAME}[] = R\"SHD(
${GLSL_CONTENT}
)SHD\";
"
    )

    file(WRITE "${OUT_FILE}" "${HEADER_TEXT}")
endforeach()

# 写 stamp 文件，让 CMake 知道这个 custom_command 的 OUTPUT 已经生成
if(DEFINED STAMP_FILE)
    string(TIMESTAMP NOW "%Y-%m-%d %H:%M:%S")
    file(WRITE "${STAMP_FILE}" "generated at ${NOW}\n")
endif()
