#version 450

layout(location = 0) in vec3 uvw;
layout(binding = 1) uniform sampler3D tex;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(tex, uvw);
}
