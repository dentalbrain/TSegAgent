#include "offscreen_renderer.h"
#include <Eigen/Eigen>
#include <utility>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

int Renderer::init_gl_wnd_program() {
    auto eglQueryDevicesEXT = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
    auto eglGetPlatformDisplayEXT = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));

    if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT) {
        std::cerr << "EGL_EXT_device_base / EGL_EXT_platform_base not supported" << std::endl;
        return -1;
    }

    EGLDeviceEXT devices[8];
    EGLint numDevices = 0;

    if (!eglQueryDevicesEXT(8, devices, &numDevices) || numDevices == 0) {
        std::cerr << "No EGLDevice found" << std::endl;
        return -1;
    }

    // 这里简单起见选第一个 GPU，你也可以根据需要挑其它 device
    EGLDeviceEXT device = devices[0];

    display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                       device,
                                       nullptr);  // 一般 attrs 为空即可
    if (display == EGL_NO_DISPLAY) {
        std::cerr << "Failed to get EGL display from EGLDevice" << std::endl;
        return -1;
    }

//    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
//    if (display == EGL_NO_DISPLAY) {
//        std::cerr << "Failed to get EGL display" << std::endl;
//        return -1;
//    }

    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        std::cerr << "Failed to initialize EGL" << std::endl;
        return -1;
    }

    // 选择 EGL 配置
    const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RED_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, 24,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
            EGL_NONE
    };

    EGLint numConfigs;
    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs)) {
        std::cerr << "Failed to choose EGL config" << std::endl;
        return -1;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::cerr << "Failed to bind OpenGL API" << std::endl;
        return -1;
    }

    // 创建 PBuffer 表面
    const EGLint surfaceAttribs[] = {
            EGL_WIDTH, width,
            EGL_HEIGHT, height,
            EGL_NONE
    };
    surface = eglCreatePbufferSurface(display, config, surfaceAttribs);
    if (surface == EGL_NO_SURFACE) {
        std::cerr << "Failed to create EGL surface" << std::endl;
        return -1;
    }

    // 创建 EGL 上下文
    const EGLint contextAttribs[] = {
            EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL_NONE
    };
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        std::cerr << "Failed to create EGL context" << std::endl;
        return -1;
    }

    // 使上下文成为当前上下文
    if (!eglMakeCurrent(display, surface, surface, context)) {
        std::cerr << "Failed to make EGL context current" << std::endl;
        return -1;
    }

    // 初始化 GLAD
    if (!gladLoadGLLoader((GLADloadproc) eglGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    return 0;
}

void Renderer::destroy() {
    if (display != EGL_NO_DISPLAY) {
        if (context != EGL_NO_CONTEXT) {
            eglDestroyContext(display, context);
        }
        if (surface != EGL_NO_SURFACE) {
            eglDestroySurface(display, surface);
        }
        eglTerminate(display);
    }
}

void Renderer::init_scene(RenderConfig render_config) {
    glViewport(0, 0, width, height);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // 启用面剔除
//    glEnable(GL_CULL_FACE);
    // 设置剔除背面，默认是 GL_BACK
//    glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);
    // 设置正面的绕序，默认是 GL_CCW（逆时针）
    glFrontFace(GL_CCW);

    glClearColor(render_config.backgroundColor[0], render_config.backgroundColor[1], render_config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::render_mesh(std::vector<std::vector<double>> vertices, std::vector<std::vector<uint32_t>> faces, RenderConfig render_config) {
    Eigen::MatrixX3d mVertices;
    mVertices.resize((long) vertices.size(), 3);
    for (int i = 0; i < vertices.size(); i++) {
        for (int j = 0; j < 3; j++) {
            mVertices(i, j) = vertices[i][j];
        }
    }

//    glEnable(GL_DEPTH_TEST)
    glClearColor(0.4f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

//    glBegin(GL_TRIANGLES);
//    glColor3f(0.5f, 0.1f, 0.2f);
//    for (auto &face: faces) {
//        for (unsigned int vi: face) {
//            glVertex3d(mVertices(vi, 0), mVertices(vi, 1), mVertices(vi, 2));
//        }
//    }
//
//    glEnd();
//    glfwSwapBuffers(window);
}

void Renderer::loop() {

}

void Renderer::set_camera(std::vector<float> r, std::vector<float> t, double z, double n, double f) {
    camera_rotate = std::move(r);
    camera_translate = std::move(t);
    zoom = z;
    near = n;
    far = f;

    proj = glm::ortho(-width / 2.0 / zoom, width / 2.0 / zoom, -height / 2.0 / zoom, height / 2.0 / zoom, near, far);
}

void Renderer::set_height(int h) {
    height = h;
}

void Renderer::set_width(int w) {
    width = w;
}

void Renderer::render_mesh(Mesh &mesh, Shader shader, RenderConfig render_config) {
    shader.Use();

    glm::mat4 rot(1);
    rot = glm::rotate(rot, glm::radians(camera_rotate[0]), glm::vec3(1, 0, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[1]), glm::vec3(0, 1, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[2]), glm::vec3(0, 0, 1));
    rot = glm::inverse(rot);
    glm::mat4 trans(1);
    trans = glm::translate(trans, glm::vec3(-camera_translate[0], -camera_translate[1], -camera_translate[2]));
    trans = rot * trans;

    glm::mat4 model(1);
    model = glm::rotate(model, glm::radians(render_config.R[0]), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(render_config.R[1]), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(render_config.R[2]), glm::vec3(0, 0, 1));
    model = glm::translate(model, glm::vec3(render_config.T[0], render_config.T[1], render_config.T[2]));

    GLint modelLoc = glGetUniformLocation(shader.Program, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    GLint viewLoc = glGetUniformLocation(shader.Program, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(trans));

    GLint projLoc = glGetUniformLocation(shader.Program, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));

    glm::vec3 lightPos = {camera_translate[0], camera_translate[1], camera_translate[2]};
    glUniform3f(glGetUniformLocation(shader.Program, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
    glUniform3f(glGetUniformLocation(shader.Program, "lightColor"), 1.0, 1.0, 1.0);
    glUniform3f(glGetUniformLocation(shader.Program, "viewPos"), camera_translate[0], camera_translate[1],
                camera_translate[2]);

    glUniform1i(glGetUniformLocation(shader.Program, "vertexColors"), render_config.vertexColors ? 1 : 0);
    glUniform4f(glGetUniformLocation(shader.Program, "vertexColor"), render_config.vertexColor[0], render_config.vertexColor[1], render_config.vertexColor[2], 1.0f);
    glUniform1f(glGetUniformLocation(shader.Program, "uAmbientStrength"), render_config.ambientStrength);
    glUniform1f(glGetUniformLocation(shader.Program, "uDiffuseStrength"), render_config.diffuseStrength);
    glUniform1f(glGetUniformLocation(shader.Program, "uSpecularStrength"), render_config.specularStrength);

    glClearColor(render_config.backgroundColor[0], render_config.backgroundColor[1], render_config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mesh.tri_draw(width, height);
}

void Renderer::render_mesh_normal(Mesh &mesh, RenderConfig render_config) {
    Shader shader(1);
    shader.Use();

    glm::mat4 rot(1);
    rot = glm::rotate(rot, glm::radians(camera_rotate[0]), glm::vec3(1, 0, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[1]), glm::vec3(0, 1, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[2]), glm::vec3(0, 0, 1));
    rot = glm::inverse(rot);
    glm::mat4 trans(1);
    trans = glm::translate(trans, glm::vec3(-camera_translate[0], -camera_translate[1], -camera_translate[2]));
    trans = rot * trans;

    glm::mat4 model(1);
    model = glm::rotate(model, glm::radians(render_config.R[0]), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(render_config.R[1]), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(render_config.R[2]), glm::vec3(0, 0, 1));
    model = glm::translate(model, glm::vec3(render_config.T[0], render_config.T[1], render_config.T[2]));

    GLint modelLoc = glGetUniformLocation(shader.Program, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    GLint viewLoc = glGetUniformLocation(shader.Program, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(trans));

    GLint projLoc = glGetUniformLocation(shader.Program, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));

    glClearColor(render_config.backgroundColor[0], render_config.backgroundColor[1], render_config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mesh.tri_draw(width, height);
}

void Renderer::render_mesh_depth(Mesh &mesh, RenderConfig render_config) {
    Shader shader(2);
    shader.Use();

    glm::mat4 rot(1);
    rot = glm::rotate(rot, glm::radians(camera_rotate[0]), glm::vec3(1, 0, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[1]), glm::vec3(0, 1, 0));
    rot = glm::rotate(rot, glm::radians(camera_rotate[2]), glm::vec3(0, 0, 1));
    rot = glm::inverse(rot);
    glm::mat4 trans(1);
    trans = glm::translate(trans, glm::vec3(-camera_translate[0], -camera_translate[1], -camera_translate[2]));
    trans = rot * trans;

    glm::mat4 model(1);
    model = glm::rotate(model, glm::radians(render_config.R[0]), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(render_config.R[1]), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(render_config.R[2]), glm::vec3(0, 0, 1));
    model = glm::translate(model, glm::vec3(render_config.T[0], render_config.T[1], render_config.T[2]));

    GLint modelLoc = glGetUniformLocation(shader.Program, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    GLint viewLoc = glGetUniformLocation(shader.Program, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(trans));

    GLint projLoc = glGetUniformLocation(shader.Program, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));

    float minDepth = FLT_MAX;
    float maxDepth = -FLT_MAX;

    for (const auto& vertex : mesh.vertices) {
        glm::vec4 projPos = proj * trans * glm::vec4(vertex.position, 1.0f);
        float viewDepth = projPos.z / projPos.w;
        minDepth = glm::min(minDepth, viewDepth);
        maxDepth = glm::max(maxDepth, viewDepth);
    }

    // 将 minDepth 和 maxDepth 设置到着色器中
    GLint minDepthLoc = glGetUniformLocation(shader.Program, "minDepth");
    GLint maxDepthLoc = glGetUniformLocation(shader.Program, "maxDepth");
    glUniform1f(minDepthLoc, minDepth);
    glUniform1f(maxDepthLoc, maxDepth);

    glClearColor(render_config.backgroundColor[0], render_config.backgroundColor[1], render_config.backgroundColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mesh.tri_draw(width, height);
}

// 读取像素的三角形 ID
std::vector<std::vector<int>> Renderer::read_triangle_id(Mesh& mesh) {
    std::vector<std::vector<int>> triangleIDs(height, std::vector<int>(width));
    GLint pixelData[width * height];
    glBindTexture(GL_TEXTURE_2D, mesh.idTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_INT, pixelData);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // OpenGL 的原点在左下角，需要翻转 Y 轴存储数据
            triangleIDs[height - 1 - y][x] = pixelData[y * width + x];
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    return triangleIDs;
}

std::vector<std::vector<std::vector<int>>> Renderer::read_image(Mesh& mesh) {
    std::vector<std::vector<std::vector<int>>> image(height, std::vector<std::vector<int>>(width, std::vector<int>(3)));
    GLubyte pixelData[width * height * 4];
    // 读取 RGB 数据
    glBindTexture(GL_TEXTURE_2D, mesh.colorTexture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelData);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                // OpenGL 的原点在左下角，需要翻转 Y 轴存储数据
                image[height - 1 - y][x][c] = pixelData[(y * width + x) * 4 + c];
            }
        }
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    return image;
}