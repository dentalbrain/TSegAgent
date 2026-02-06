#ifndef REVERSIBLE_RASTERIZER_SHADER_H
#define REVERSIBLE_RASTERIZER_SHADER_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#include "glspec.h"

struct ShaderProgram {
    GLint length;
    GLenum format;
    std::vector<char> binary;
    bool valid = false;
};

class Shader
{
public:
    Shader(int type = 0);

// 程序ID
    GLuint Program;
    // 构造器读取并构建着色器
    Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
    // 使用程序
    void Use();
    static ShaderProgram* get_cached_programs() {
        static ShaderProgram cached[3];
        return cached;
    }
private:
    std::string get_vertex_shader(int type);
    std::string get_fragment_shader(int type);
};

#endif //REVERSIBLE_RASTERIZER_SHADER_H
