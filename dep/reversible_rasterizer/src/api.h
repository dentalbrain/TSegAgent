#ifndef REVERSIBLE_RASTERIZER_API_H
#define REVERSIBLE_RASTERIZER_API_H

#ifdef USE_OFFSCREEN_RENDERER
#include "offscreen_renderer.h"
#else
#include "renderer.h"
#endif

#include "config.h"
#include "mesh.hpp"

Renderer *init();

void setup_mesh(Mesh &mesh, CameraConfig config);

std::vector<std::vector<int>> render_id_map(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config);

std::vector<std::vector<std::vector<int>>> render(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config);

std::vector<std::vector<std::vector<int>>> render_normal(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config);

std::vector<std::vector<std::vector<int>>> render_depth(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config);

std::vector<std::vector<std::vector<int>>> render_stacked(Renderer *renderer, Mesh &mesh, CameraConfig config, RenderConfig render_config, int stack_mask, const std::vector<float> &stack_alphas);

void end(Renderer *renderer);

void end_with_mesh(Renderer *renderer, Mesh &mesh);

#endif // REVERSIBLE_RASTERIZER_API_H
