use watertender::prelude::*;
use watertender::memory::UsageFlags;
use defaults::FRAMES_IN_FLIGHT;
use anyhow::Result;
use crate::{Input, RenderSettings};
use std::ffi::CString;

struct App {
    quad_mesh: ManagedMesh,
    instances: ManagedBuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    cfg: RenderSettings,
    input: Input,

    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,

    scene_ubo: FrameDataUbo<SceneData>,
    camera: MultiPlatformCamera,
    anim: f32,
    starter_kit: StarterKit,
}

const TEX_IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

type RenderInputs = (Input, RenderSettings);

pub fn visualize(input: Input, cfg: RenderSettings, vr: bool) -> Result<()> {
    crate::verify_input(&input, &cfg)?;
    let info = AppInfo::default().validation(true);
    launch::<App, RenderInputs>(info, vr, (input, cfg))
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SceneData {
    pub cameras: [f32; 4 * 4 * 2],
    pub anim: f32,
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QuadInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub layer: f32,
}

unsafe impl bytemuck::Zeroable for QuadInstance {}
unsafe impl bytemuck::Pod for QuadInstance {}

impl QuadInstance {
    pub fn new([x, y, z]: [f32; 3], layer: f32) -> Self {
        Self { x, y, z, layer }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescriptionBuilder<'static> {
        vk::VertexInputBindingDescriptionBuilder::new()
            .binding(1)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)
    }

    pub fn get_attribute_descriptions() -> vk::VertexInputAttributeDescriptionBuilder<'static> {
        vk::VertexInputAttributeDescriptionBuilder::new()
            .binding(1)
            .location(2)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0)
    }
}

impl App {
    pub fn upload(&mut self, idx: usize) -> Result<()> {
        let points_stride = (self.cfg.input_points * 4) as usize;
        let points = &self.input.points[points_stride*idx..points_stride*(idx+1)];

        // Upload to instance buffer
        self.instances.write_bytes(0, &bytemuck::cast_slice(&points))?;

        Ok(())
    }
}

impl MainLoop<RenderInputs> for App {
    fn new(core: &SharedCore, mut platform: Platform<'_>, (input, cfg): RenderInputs) -> Result<Self> {
        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Create instance buffer
        let instance_buffer_size = (cfg.input_points * 4) as usize * std::mem::size_of::<f32>();
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(instance_buffer_size as _);

        let mut instances = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
        )?;

        // Create image staging buffer
        let image_buffer_size = cfg.input_images * cfg.input_height * cfg.input_width * 4;
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(image_buffer_size as _);

        let mut image_staging_buffer = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
        )?;

        // Create image cube
        let image_cube_extent = vk::Extent3DBuilder::new()
            .width(cfg.input_width)
            .height(cfg.input_height)
            .depth(cfg.input_images)
            .build();
        let ci = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .format(TEX_IMAGE_FORMAT)
            .extent(image_cube_extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlagBits::_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?;

        // Create image views
        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .build();

        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image.instance())
            .view_type(vk::ImageViewType::_3D)
            .format(TEX_IMAGE_FORMAT)
            .subresource_range(subresource_range)
            .build();

        let image_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        // Create sampler
        let create_info = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(16.)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(0.)
            .build();

        let sampler = unsafe { core.device.create_sampler(&create_info, None, None) }.result()?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Scene data
        let scene_ubo = FrameDataUbo::new(core.clone(), FRAMES_IN_FLIGHT)?;

        // Create descriptor set layout
        const FRAME_DATA_BINDING: u32 = 0;
        const TEX_DATA_BINDING: u32 = 1;
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(FRAME_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(TEX_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        ];

        let descriptor_set_layout_ci =
            vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            core.device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None, None)
        }
        .result()?;

        // Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(FRAMES_IN_FLIGHT as _),
        ];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_sizes)
            .max_sets((FRAMES_IN_FLIGHT * 2) as _);

        let descriptor_pool =
            unsafe { core.device.create_descriptor_pool(&create_info, None, None) }.result()?;

        // Create descriptor sets
        let layouts = vec![descriptor_set_layout; FRAMES_IN_FLIGHT];
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);

        let descriptor_sets =
            unsafe { core.device.allocate_descriptor_sets(&create_info) }.result()?;

        // Image info
        let image_infos = [vk::DescriptorImageInfoBuilder::new()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(image_view)
            .sampler(sampler)];

        // Write descriptor sets
        for (frame, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let frame_data_bi = [scene_ubo.descriptor_buffer_info(frame)];
            let writes = [
                vk::WriteDescriptorSetBuilder::new()
                    .buffer_info(&frame_data_bi)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .dst_set(descriptor_set)
                    .dst_binding(FRAME_DATA_BINDING)
                    .dst_array_element(0),
                vk::WriteDescriptorSetBuilder::new()
                    .image_info(&image_infos)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_set(descriptor_set)
                    .dst_binding(TEX_DATA_BINDING)
                    .dst_array_element(0)
            ];

            unsafe {
                core.device.update_descriptor_sets(&writes, &[]);
            }
        }

        let descriptor_set_layouts = [descriptor_set_layout];

        // Pipeline layout
        let push_constant_ranges = [vk::PushConstantRangeBuilder::new()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<[f32; 4 * 4]>() as u32)];

        let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { core.device.create_pipeline_layout(&create_info, None, None) }.result()?;

        // Pipeline
        let pipeline = shader(
            core,
            &std::fs::read("shaders/unlit.vert.spv")?,
            &std::fs::read("shaders/unlit.frag.spv")?,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            starter_kit.render_pass,
            pipeline_layout,
        )?;

        // Mesh uploads
        let (vertices, indices) = quad(1.);
        let rainbow_cube = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        let mut app = Self {
            cfg,
            input,
            instances,
            camera,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            anim: 0.0,
            pipeline_layout,
            scene_ubo,
            quad_mesh: rainbow_cube,
            pipeline,
            starter_kit,
        };

        app.upload(0)?;

        Ok(app)
    }

    fn frame(
        &mut self,
        frame: Frame,
        core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;

        unsafe {
            core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.starter_kit.frame]],
                &[],
            );

            // Draw cmds
            core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            core.device.cmd_bind_index_buffer(
                command_buffer,
                self.quad_mesh.indices.instance(),
                0,
                vk::IndexType::UINT32,
            );

            core.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.quad_mesh.vertices.instance()],
                &[0]
            );

            core.device.cmd_bind_vertex_buffers(
                command_buffer,
                1,
                &[self.instances.instance()],
                &[0]
            );

            core.device.cmd_draw_indexed(
                command_buffer,
                self.quad_mesh.n_indices,
                self.cfg.input_points,
                0,
                0,
                0,
            );

        }

        let (ret, cameras) = self.camera.get_matrices(&platform)?;

        self.scene_ubo.upload(
            self.starter_kit.frame,
            &SceneData {
                cameras,
                anim: self.anim,
            },
        )?;

        // End draw cmds
        self.starter_kit.end_command_buffer(cmd)?;

        Ok(ret)
    }

    fn swapchain_resize(&mut self, images: Vec<vk::Image>, extent: vk::Extent2D) -> Result<()> {
        self.starter_kit.swapchain_resize(images, extent)
    }

    fn event(
        &mut self,
        mut event: PlatformEvent<'_, '_>,
        _core: &Core,
        mut platform: Platform<'_>,
    ) -> Result<()> {
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl SyncMainLoop<RenderInputs> for App {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}

fn quad(size: f32) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        Vertex::new([-size, -size, 0.], [0., 0., 0.]),
        Vertex::new([-size, size, 0.], [0., 1., 0.]),
        Vertex::new([size, size, 0.], [1., 1., 0.]),
        Vertex::new([size, -size, 0.], [1., 0., 0.]),
    ];

    let indices = vec![
        0, 1, 2,
        0, 2, 3,
    ];

    (vertices, indices)
}

pub fn shader(
    prelude: &Core,
    vertex_src: &[u8],
    fragment_src: &[u8],
    primitive: vk::PrimitiveTopology,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    // Create shader modules
    let vert_decoded = erupt::utils::decode_spv(vertex_src)?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
    let vertex = unsafe {
        prelude
            .device
            .create_shader_module(&create_info, None, None)
    }
    .result()?;

    let frag_decoded = erupt::utils::decode_spv(fragment_src)?;
    let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
    let fragment = unsafe {
        prelude
            .device
            .create_shader_module(&create_info, None, None)
    }
    .result()?;

    let vert_attrib_desc = Vertex::get_attribute_descriptions();
    let inst_attrib_desc = QuadInstance::get_attribute_descriptions();
    let attribute_descriptions = [
        vert_attrib_desc[0],
        vert_attrib_desc[1],
        inst_attrib_desc,
    ];
    let binding_descriptions = [
        Vertex::binding_description(),
        QuadInstance::binding_description(),
    ];

    // Build pipeline
    let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
        .vertex_attribute_descriptions(&attribute_descriptions[..])
        .vertex_binding_descriptions(&binding_descriptions);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(primitive)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
        .viewport_count(1)
        .scissor_count(1);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfoBuilder::new().dynamic_states(&dynamic_states);

    let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_clamp_enable(false);

    let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(true)
        .color_blend_op(vk::BlendOp::ADD)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
    ];
    let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);

    let entry_point = CString::new("main")?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(vertex)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(fragment)
            .name(&entry_point),
    ];

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let create_info = vk::GraphicsPipelineCreateInfoBuilder::new()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .depth_stencil_state(&depth_stencil_state)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = unsafe {
        prelude
            .device
            .create_graphics_pipelines(None, &[create_info], None)
    }
    .result()?[0];

    unsafe {
        prelude.device.destroy_shader_module(Some(fragment), None);
        prelude.device.destroy_shader_module(Some(vertex), None);
    }

    Ok(pipeline)
}
