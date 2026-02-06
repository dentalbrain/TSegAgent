#ifndef REVERSIBLE_RASTERIZER_RENDERER_H
#define REVERSIBLE_RASTERIZER_RENDERER_H

#define WIDTH 1278
#define HEIGHT 798

#ifdef USE_OFFSCREEN_RENDERER
#include "glspec.h"
#include <vector>
#include "config.h"
#include "mesh.hpp"
#include "shader.h"

class Renderer {
public:
    int init_gl_wnd_program();
    void render_mesh(std::vector<std::vector<double>>, std::vector<std::vector<uint32_t>>, RenderConfig render_config);
    void render_mesh(Mesh &mesh, Shader shader, RenderConfig render_config);
    void render_mesh_normal(Mesh &mesh, RenderConfig render_config);
    void render_mesh_depth(Mesh &mesh, RenderConfig render_config);
    void init_scene(RenderConfig render_config);
    void destroy();
    void loop();

    void set_width(int);
    void set_height(int);
    void set_camera(std::vector<float> r, std::vector<float> t, double zoom, double near, double far);

    std::vector<std::vector<int>> read_triangle_id(Mesh &mesh);
    std::vector<std::vector<std::vector<int>>> read_image(Mesh &mesh);

    Renderer() : width(WIDTH), height(HEIGHT) {
    }

    ~Renderer() {
    }

//    GLubyte* buffer;
private:
    int width, height;
    double near, far;
    double zoom;
    std::vector<float> camera_rotate;
    std::vector<float> camera_translate;
    glm::mat4 proj;

    EGLDisplay display;
    EGLConfig config;
    EGLSurface surface;
    EGLContext context;
};

#endif
#endif //REVERSIBLE_RASTERIZER_RENDERER_H
