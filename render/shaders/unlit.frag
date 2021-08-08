#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 uvw;
layout(binding = 1) uniform sampler3D tex;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(tex, uvw);
}
