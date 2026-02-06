#ifndef REVERSIBLE_RASTERIZER_GLSPEC_H
#define REVERSIBLE_RASTERIZER_GLSPEC_H

#ifdef USE_OFFSCREEN_RENDERER
#include <glad/glad.h>
#ifndef APIENTRY
#define APIENTRY GLAPIENTRY
#endif
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>

#else
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#endif

#endif //REVERSIBLE_RASTERIZER_GLSPEC_H
