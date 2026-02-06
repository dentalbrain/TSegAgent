#include "api.h"

Renderer *init()
{
    Renderer *renderer = new Renderer();
    if (renderer->init_gl_wnd_program() != 0)
    {
        std::cout << "Renderer init failed" << std::endl;
        renderer->destroy();
        delete renderer;
        return nullptr;
    }
    return renderer;
}

void setup_mesh(Mesh &mesh, CameraConfig config)
{
    mesh.transform_center();
    mesh.setup_tri_mesh(config.width, config.height);
}

std::vector<std::vector<int>> render_id_map(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config)
{
    Shader shader(0);
    renderer->set_width(config.width);
    renderer->set_height(config.height);
    renderer->set_camera(
        config.R,
        config.T,
        config.zoom,
        config.near,
        config.far);
    // mesh.transform_center();

    renderer->init_scene(render_config);
    // mesh.setup_tri_mesh(config.width, config.height);
    renderer->render_mesh(mesh, shader, render_config);
    std::vector<std::vector<int>> id_map = renderer->read_triangle_id(mesh);
    // mesh.destroy();
    //    renderer->destroy();
    //    delete renderer;
    return id_map;
}

std::vector<std::vector<std::vector<int>>> render(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config)
{
    Shader shader(0);
    renderer->set_width(config.width);
    renderer->set_height(config.height);
    renderer->set_camera(
        config.R,
        config.T,
        config.zoom,
        config.near,
        config.far);
    // mesh.transform_center();

    renderer->init_scene(render_config);
    // mesh.setup_tri_mesh(config.width, config.height);
    renderer->render_mesh(mesh, shader, render_config);
    std::vector<std::vector<std::vector<int>>> img = renderer->read_image(mesh);
    // mesh.destroy();
    return img;
}

std::vector<std::vector<std::vector<int>>> render_normal(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config)
{
    renderer->set_width(config.width);
    renderer->set_height(config.height);
    renderer->set_camera(
        config.R,
        config.T,
        config.zoom,
        config.near,
        config.far);
    // mesh.transform_center();

    renderer->init_scene(render_config);
    // mesh.setup_tri_mesh(config.width, config.height);
    renderer->render_mesh_normal(mesh, render_config);
    std::vector<std::vector<std::vector<int>>> img = renderer->read_image(mesh);
    // mesh.destroy();
    return img;
}

std::vector<std::vector<std::vector<int>>> render_depth(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config)
{
    renderer->set_width(config.width);
    renderer->set_height(config.height);
    renderer->set_camera(
        config.R,
        config.T,
        config.zoom,
        config.near,
        config.far);
    // mesh.transform_center();

    renderer->init_scene(render_config);
    // mesh.setup_tri_mesh(config.width, config.height);
    renderer->render_mesh_depth(mesh, render_config);
    std::vector<std::vector<std::vector<int>>> img = renderer->read_image(mesh);
    // mesh.destroy();
    return img;
}

inline void accumulate(
    std::vector<std::vector<std::vector<float>>> &dst,
    const std::vector<std::vector<std::vector<int>>> &src,
    float alpha)
{
    int W = dst.size();
    int H = dst[0].size();

    for (int w = 0; w < W; ++w)
        for (int h = 0; h < H; ++h)
            for (int c = 0; c < 3; ++c)
                dst[w][h][c] += alpha * src[w][h][c];
}

std::vector<std::vector<std::vector<int>>>
render_stacked(
    Renderer *renderer,
    Mesh &mesh,
    CameraConfig config,
    RenderConfig render_config,
    int stack_mask,
    const std::vector<float> &stack_alphas)
{
    if (stack_mask == 0)
        return {};

    int cnt = __builtin_popcount(stack_mask);
    if (cnt != stack_alphas.size())
    {
        std::cerr << "Stack mask and alphas size mismatch\n";
        return {};
    }

    renderer->set_width(config.width);
    renderer->set_height(config.height);
    renderer->set_camera(
        config.R, config.T, config.zoom, config.near, config.far);
    renderer->init_scene(render_config);

    // 累加用 float
    std::vector<std::vector<std::vector<float>>> stacked_image(
        config.height,
        std::vector<std::vector<float>>(
            config.width,
            std::vector<float>(3, 0.0f)));

    int alpha_idx = 0;

    if (stack_mask & 0x1)
    {
        Shader shader(0);
        renderer->render_mesh(mesh, shader, render_config);
        auto img = renderer->read_image(mesh);
        accumulate(stacked_image, img, stack_alphas[alpha_idx++]);
    }

    if (stack_mask & 0x2)
    {
        renderer->render_mesh_normal(mesh, render_config);
        auto img = renderer->read_image(mesh);
        accumulate(stacked_image, img, stack_alphas[alpha_idx++]);
    }

    if (stack_mask & 0x4)
    {
        renderer->render_mesh_depth(mesh, render_config);
        auto img = renderer->read_image(mesh);
        accumulate(stacked_image, img, stack_alphas[alpha_idx++]);
    }

    // float → int
    std::vector<std::vector<std::vector<int>>> result(
        config.height,
        std::vector<std::vector<int>>(
            config.width,
            std::vector<int>(3)));

    for (int h = 0; h < config.height; ++h)
        for (int w = 0; w < config.width; ++w)
            for (int c = 0; c < 3; ++c)
                result[h][w][c] =
                    std::clamp<int>(static_cast<int>(stacked_image[h][w][c]), 0, 255);

    return result;
}

void end(Renderer *renderer)
{
    if (renderer != nullptr)
    {
        renderer->destroy();
        delete renderer;
    }
}

void end_with_mesh(Renderer *renderer, Mesh &mesh)
{
    if (renderer != nullptr)
    {
        renderer->destroy();
        delete renderer;
    }
    mesh.destroy();
}