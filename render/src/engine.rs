use watertender::prelude::*;
use watertender::memory::UsageFlags;
use defaults::FRAMES_IN_FLIGHT;
use anyhow::Result;
use crate::{Input, RenderSettings};
use std::ffi::CString;

pub struct Engine {
    cfg: RenderSettings,
    quad_mesh: ManagedMesh,
    instances: ManagedBuffer,
    image_staging_buffer: ManagedBuffer,
    image: ManagedImage,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    image_extent: vk::Extent3D,
    scene_ubo: FrameDataUbo<SceneData>,
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pub core: SharedCore,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SceneData {
    pub cameras: [f32; 4 * 4 * 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QuadInstance {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub layer: f32,
}

const TEX_IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

impl Engine {
    pub fn new(
        core: SharedCore, 
        cfg: RenderSettings, 
        render_pass: vk::RenderPass,
        command_buffer: vk::CommandBuffer
    ) -> Result<Self> {
        // Create staging buffer
        let mut staging_buffer = StagingBuffer::new(core.clone())?;

        // Create instance buffer
        let instance_buffer_size = (cfg.input_points * 4) as usize * std::mem::size_of::<f32>();
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(instance_buffer_size as _);

        let instances = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
        )?;

        // Create image staging buffer
        let image_buffer_size = cfg.input_images * cfg.input_height * cfg.input_width * 4;
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .size(image_buffer_size as _);

        let image_staging_buffer = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
        )?;

        // Create image cube
        let image_extent = vk::Extent3DBuilder::new()
            .width(cfg.input_width)
            .height(cfg.input_height)
            .depth(cfg.input_images)
            .build();
        let ci = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_3D)
            .format(TEX_IMAGE_FORMAT)
            .extent(image_extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlagBits::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
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
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(16.)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .mip_lod_bias(0.)
            .min_lod(0.)
            .max_lod(0.)
            .build();

        let sampler = unsafe { core.device.create_sampler(&create_info, None, None) }.result()?;

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
            &core,
            include_bytes!("shaders/unlit.vert.spv"),
            include_bytes!("shaders/unlit.frag.spv"),
            vk::PrimitiveTopology::TRIANGLE_LIST,
            render_pass,
            pipeline_layout,
        )?;

        // Mesh uploads
        let (vertices, indices) = quad(0.25);
        let quad_mesh = upload_mesh(
            &mut staging_buffer,
            command_buffer,
            &vertices,
            &indices,
        )?;


        Ok(Self {
            image_extent,
            image,
            cfg,
            image_staging_buffer,
            instances,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            pipeline_layout,
            scene_ubo,
            quad_mesh,
            pipeline,
            core,
        })
    }

    pub fn write_commands(&mut self, command_buffer: vk::CommandBuffer, frame: usize, cameras: [f32; 4 * 4 * 2]) -> Result<()> {
        self.scene_ubo.upload(
            frame,
            &SceneData {
                cameras,
            },
        )?;

        unsafe {
            self.core.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame]],
                &[],
            );

            // Draw cmds
            self.core.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.core.device.cmd_bind_index_buffer(
                command_buffer,
                self.quad_mesh.indices.instance(),
                0,
                vk::IndexType::UINT32,
            );

            self.core.device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.quad_mesh.vertices.instance()],
                &[0]
            );

            self.core.device.cmd_bind_vertex_buffers(
                command_buffer,
                1,
                &[self.instances.instance()],
                &[0]
            );

            self.core.device.cmd_draw_indexed(
                command_buffer,
                self.quad_mesh.n_indices,
                self.cfg.input_points,
                0,
                0,
                0,
            );

        }

        Ok(())
    }

    pub fn upload(&mut self, idx: usize, input: &Input) -> Result<()> {
        // Upload points
        let points_batch_stride = (self.cfg.input_points * 4) as usize;
        let points = &input.points[points_batch_stride*idx..points_batch_stride*(idx+1)];

        self.instances.write_bytes(0, &bytemuck::cast_slice(&points))?;

        // Upload images
        let image_batch_stride = (self.cfg.input_images * self.cfg.input_height * self.cfg.input_width * 4) as usize;
        let image_data = &input.images[image_batch_stride*idx..image_batch_stride*(idx+1)];

        self.image_staging_buffer.write_bytes(0, bytemuck::cast_slice(image_data))?;

        Ok(())
    }

    pub fn prepare(&mut self, command_buffer: vk::CommandBuffer) -> Result<()> {
        // Record command buffer to upload to gpu_buffer
        unsafe {
            self
                .core
                .device
                .reset_command_buffer(command_buffer, None)
                .result()?;
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self
                .core
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .result()?;


            // Barrier (UNDEFINED -> TRANSFER_DST_OPTIMAL)
            let image_subresource = vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(self.image.instance())
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(image_subresource);

            self.core.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                None,
                &[],
                &[],
                &[barrier],
            );

            // Image copy
            let sub_layers = vk::ImageSubresourceLayersBuilder::new()
                .layer_count(1)
                .base_array_layer(0)
                .mip_level(0)
                .aspect_mask(vk::ImageAspectFlags::COLOR);

            let buffer_image_copy = vk::BufferImageCopyBuilder::new()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_extent(self.image_extent)
                .image_offset(vk::Offset3DBuilder::new().x(0).y(0).z(0).build())
                .image_subresource(*sub_layers);

            self.core.device.cmd_copy_buffer_to_image(
                command_buffer,
                self.image_staging_buffer.instance(),
                self.image.instance(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[buffer_image_copy]
            );

            // Barrier (TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL)
            let image_subresource = vk::ImageSubresourceRangeBuilder::new()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(self.image.instance())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(image_subresource);

            self.core.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                None,
                &[],
                &[],
                &[barrier],
            );


            self
                .core
                .device
                .end_command_buffer(command_buffer)
                .result()?;
            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self
                .core
                .device
                .queue_submit(self.core.queue, &[submit_info], None)
                .result()?;
            self
                .core.device.queue_wait_idle(self.core.queue).result()?;
        }

        Ok(())
    }

    pub fn cfg(&self) -> RenderSettings {
        self.cfg
    }
}

unsafe impl bytemuck::Zeroable for SceneData {}
unsafe impl bytemuck::Pod for SceneData {}

unsafe impl bytemuck::Zeroable for QuadInstance {}
unsafe impl bytemuck::Pod for QuadInstance {}

impl QuadInstance {
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
        .dst_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
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

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_descriptor_pool(Some(self.descriptor_pool), None);
            self.core.device.destroy_descriptor_set_layout(Some(self.descriptor_set_layout), None);
        }
    }
}
