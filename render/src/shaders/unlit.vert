#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_multiview : require

// Camera UBO
layout(set = 0, binding = 0) uniform Animation {
    mat4 camera[2];
};

// From vertex buffer
layout(location = 0) in vec3 vert_pos;
layout(location = 1) in vec3 vert_color;

// From instance buffer
layout(location = 2) in vec4 instance;

// To fragment shader
layout(location = 0) out vec3 frag_inputs;

void main() {
    vec4 screen_pos = camera[gl_ViewIndex] * vec4(instance.xyz, 1.0);
    gl_Position = screen_pos + vec4(vert_pos, 0.);
    frag_inputs = vec3(vert_color.xy, instance.w);
}
