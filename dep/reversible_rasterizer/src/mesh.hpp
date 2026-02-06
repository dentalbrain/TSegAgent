#ifndef REVERSIBLE_RASTERIZER_MESH_HPP
#define REVERSIBLE_RASTERIZER_MESH_HPP

#include "glspec.h"

#include <utility>
#include <vector>
#include <cstdint>
#include <glm/matrix.hpp>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <fstream>
#include <assimp/scene.h>
#include <assimp/Exporter.hpp>

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 color;
};

struct TriVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 color;
    glm::int32 triangleId;
};

struct Texture {
    GLuint id;
    std::string type;
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<TriVertex> triVertices;
    std::vector<GLuint> indices;
    Eigen::MatrixX3d mVertices;
    Eigen::MatrixX3d mNormals;
    Eigen::MatrixX4d mVertexColors;
    Eigen::MatrixX3i mFaces;
    GLuint idTexture = 0, colorTexture = 0;

    Mesh(std::vector<std::vector<double>> v, std::vector<std::vector<double>> n, std::vector<std::vector<uint32_t>> f)
            : Mesh(std::move(v), std::move(n), std::move(f), {}) {}

    Mesh(std::vector<std::vector<double>> v, std::vector<std::vector<double>> n, std::vector<std::vector<uint32_t>> f,
         std::vector<std::vector<float>> vc) {
        std::cout << "Creating mesh with " << v.size() << " vertices and " << f.size() << " faces" << std::endl;
        mVertices.resize((long) v.size(), 3);
        mNormals.resize((long) v.size(), 3);
        mVertexColors.resize((long) v.size(), 4);
        for (int i = 0; i < v.size(); i++) {
            for (int j = 0; j < 3; j++) {
                mVertices(i, j) = v[i][j];
                if (!n.empty()) {
                    mNormals(i, j) = n[i][j];
                }
            }
            for (int j = 0; j < 4; j++) {
                if (!vc.empty()) {
                    mVertexColors(i, j) = vc[i][j];
                }
            }
        }

        mFaces.resize((long) f.size(), 3);
        for (int i = 0; i < f.size(); i++) {
            for (int j = 0; j < 3; j++) {
                mFaces(i, j) = (int) f[i][j];
            }
        }

        clear_cache();
        apply();
    }

    ~Mesh() {
        destroy();
    }

    void transform_center() {
        Eigen::RowVector3d bb_min = mVertices.colwise().minCoeff();
        Eigen::RowVector3d bb_max = mVertices.colwise().maxCoeff();
        Eigen::RowVector3d center = (bb_max + bb_min) / 2.0;
        mVertices.rowwise() -= center;

        clear_cache();
        apply();
    }

    void setup_tri_mesh(int width, int height) {
        glGenVertexArrays(1, &this->VAO);
        glGenBuffers(1, &this->VBO);

        glBindVertexArray(this->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, this->VBO);

        glBufferData(GL_ARRAY_BUFFER, this->triVertices.size() * sizeof(TriVertex),
                     &this->triVertices[0], GL_STATIC_DRAW);

        // 设置顶点坐标指针
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(TriVertex),
                              (GLvoid *) 0);
        // 设置法线指针
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(TriVertex),
                              (GLvoid *) offsetof(TriVertex, normal));
        // 设置顶点的纹理坐标
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(TriVertex),
                              (GLvoid *) offsetof(TriVertex, color));

        // 设置顶点的ID纹理坐标
        glEnableVertexAttribArray(3);
        glVertexAttribIPointer(3, 1, GL_INT, sizeof(TriVertex), (GLvoid *) offsetof(TriVertex, triangleId));

        glBindVertexArray(0);

        // -----------------------------------------------

        glGenFramebuffers(1, &FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);

        // 创建颜色纹理附件
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

        // 创建整数 ID 纹理附件
        glGenTextures(1, &idTexture);
        glBindTexture(GL_TEXTURE_2D, idTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, idTexture, 0);

        glGenRenderbuffers(1, &depthRBO);
        glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);

        // 设置渲染到多个目标
        GLuint attachments[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, attachments);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Framebuffer not complete!" << std::endl;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void tri_draw(int width, int height) {
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);

        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // 清 COLOR_ATTACHMENT1（R32I）为 0
        const GLint clearId[1] = {-1};
        glClearBufferiv(GL_COLOR, 1, clearId);         // 1 对应 COLOR_ATTACHMENT1

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, triVertices.size());

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        // 绑定默认帧缓冲
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // 将内容绘制到默认帧缓冲
        glBindFramebuffer(GL_READ_FRAMEBUFFER, FBO); // 读取自定义 FBO

        // 拷贝颜色附件到默认帧缓冲
        glBlitFramebuffer(
                0, 0, width, height,  // 源区域（FBO）
                0, 0, width, height,  // 目标区域（窗口）
                GL_COLOR_BUFFER_BIT, GL_NEAREST
        );
    }

    void destroy() {
        if (VAO) {
            glDeleteVertexArrays(1, &VAO);
            VAO = 0;
        }
        if (VBO) {
            glDeleteBuffers(1, &VBO);
            VBO = 0;
        }
        if (EBO) {
            glDeleteBuffers(1, &EBO);
            EBO = 0;
        }
        if (colorTexture) {
            glDeleteTextures(1, &colorTexture);
            colorTexture = 0;
        }
        if (idTexture) {
            glDeleteTextures(1, &idTexture);
            idTexture = 0;
        }
        if (FBO) {
            glDeleteFramebuffers(1, &FBO);
            FBO = 0;
        }
        if (depthRBO) {
            glDeleteRenderbuffers(1, &depthRBO);
            depthRBO = 0;
        }
    }

private:
    GLuint VAO = 0, VBO = 0, EBO = 0, FBO = 0, depthRBO = 0;

    // Disable copy to avoid double-free of GL resources
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // Allow move semantics
    Mesh(Mesh&& other) noexcept
        : vertices(std::move(other.vertices)),
          triVertices(std::move(other.triVertices)),
          indices(std::move(other.indices)),
          mVertices(std::move(other.mVertices)),
          mNormals(std::move(other.mNormals)),
          mVertexColors(std::move(other.mVertexColors)),
          mFaces(std::move(other.mFaces)),
          idTexture(other.idTexture), colorTexture(other.colorTexture),
          VAO(other.VAO), VBO(other.VBO), EBO(other.EBO), FBO(other.FBO), depthRBO(other.depthRBO) {
        other.idTexture = 0;
        other.colorTexture = 0;
        other.VAO = other.VBO = other.EBO = other.FBO = other.depthRBO = 0;
    }

    Mesh& operator=(Mesh&& other) noexcept {
        if (this != &other) {
            destroy();
            vertices = std::move(other.vertices);
            triVertices = std::move(other.triVertices);
            indices = std::move(other.indices);
            mVertices = std::move(other.mVertices);
            mNormals = std::move(other.mNormals);
            mVertexColors = std::move(other.mVertexColors);
            mFaces = std::move(other.mFaces);
            idTexture = other.idTexture;
            colorTexture = other.colorTexture;
            VAO = other.VAO; VBO = other.VBO; EBO = other.EBO; FBO = other.FBO; depthRBO = other.depthRBO;
            other.idTexture = 0;
            other.colorTexture = 0;
            other.VAO = other.VBO = other.EBO = other.FBO = other.depthRBO = 0;
        }
        return *this;
    }

    void clear_cache() {
        vertices.clear();
        triVertices.clear();
        indices.clear();
    }

    void apply() {
        if (vertices.empty() || indices.empty()) {
            vertices.resize(mVertices.rows());
            indices.resize(mFaces.size());

            for (int i = 0; i < mVertices.rows(); i++) {
                vertices[i].position = {mVertices(i, 0), mVertices(i, 1), mVertices(i, 2)};
                vertices[i].normal = {mNormals(i, 0), mNormals(i, 1), mNormals(i, 2)};
                vertices[i].color = {mVertexColors(i, 0), mVertexColors(i, 1), mVertexColors(i, 2),
                                     mVertexColors(i, 3)};
            }

            for (int i = 0; i < mFaces.rows(); i++) {
                indices[i * 3] = mFaces(i, 0);
                indices[i * 3 + 1] = mFaces(i, 1);
                indices[i * 3 + 2] = mFaces(i, 2);
            }
        }

        triVertices.resize(mFaces.size());
        for (int i = 0; i < mFaces.rows(); i++) {
            for (int j = 0; j < 3; j++) {
                int vi = mFaces(i, j);
                triVertices[i * 3 + j].position = {mVertices(vi, 0), mVertices(vi, 1), mVertices(vi, 2)};
                triVertices[i * 3 + j].normal = {mNormals(vi, 0), mNormals(vi, 1), mNormals(vi, 2)};
                triVertices[i * 3 + j].color = {mVertexColors(vi, 0), mVertexColors(vi, 1), mVertexColors(vi, 2),
                                                mVertexColors(vi, 3)};
                triVertices[i * 3 + j].triangleId = i + 1;
            }
        }
    }

};


#endif //REVERSIBLE_RASTERIZER_MESH_HPP
