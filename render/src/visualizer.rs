use watertender::prelude::*;
use watertender::memory::UsageFlags;
use defaults::FRAMES_IN_FLIGHT;
use anyhow::Result;
use crate::{Input, RenderSettings};

struct App {
    rainbow_cube: ManagedMesh,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,

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
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)
    }

    pub fn get_attribute_descriptions() -> vk::VertexInputAttributeDescriptionBuilder<'static> {
        vk::VertexInputAttributeDescriptionBuilder::new()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(0)
    }
}

impl MainLoop<RenderInputs> for App {
    fn new(core: &SharedCore, mut platform: Platform<'_>, (input, cfg): RenderInputs) -> Result<Self> {
        let mut starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        // Create instance buffer
        let instance_buffer_size = crate::points_float_count(&cfg) as usize * std::mem::size_of::<f32>();
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(instance_buffer_size as _);

        let mut instance_buffer = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
        )?;

        // Create image staging buffer
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(crate::images_byte_count(&cfg) as _);

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

        let mut images = vec![];
        for _ in 0..cfg.batch_size {
            images.push(ManagedImage::new(core.clone(), ci, UsageFlags::FAST_DEVICE_ACCESS)?);
        }

        // Create image views
        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .layer_count(1)
            .base_array_layer(0)
            .level_count(1)
            .build();

        let mut image_views = vec![];
        for image in &images {
            let create_info = vk::ImageViewCreateInfoBuilder::new()
                .image(image.instance())
                .view_type(vk::ImageViewType::_3D)
                .format(TEX_IMAGE_FORMAT)
                .subresource_range(subresource_range)
                .build();

            let image_view =
                unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;
            image_views.push(image_view);
        }

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


        // Upload to instance buffer
        instance_buffer.write_bytes(0, bytemuck::cast_slice(&input.points))?;

        // Camera
        let camera = MultiPlatformCamera::new(&mut platform);

        // Scene data
        let scene_ubo = FrameDataUbo::new(core.clone(), FRAMES_IN_FLIGHT)?;

        // Create descriptor set layout
        const FRAME_DATA_BINDING: u32 = 0;
        let bindings = [
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(FRAME_DATA_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS),
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
        let (vertices, indices) = rainbow_cube();
        let rainbow_cube = upload_mesh(
            &mut starter_kit.staging_buffer,
            starter_kit.command_buffers[0],
            &vertices,
            &indices,
        )?;

        Ok(Self {
            camera,
            descriptor_set_layout,
            descriptor_sets,
            descriptor_pool,
            anim: 0.0,
            pipeline_layout,
            scene_ubo,
            rainbow_cube,
            pipeline,
            starter_kit,
        })
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

            draw_mesh(
                core,
                command_buffer,
                &self.rainbow_cube,
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

fn rainbow_cube() -> (Vec<Vertex>, Vec<u32>) {
    let vertices = vec![
        Vertex::new([-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]),
        Vertex::new([1.0, -1.0, -1.0], [1.0, 0.0, 1.0]),
        Vertex::new([1.0, 1.0, -1.0], [1.0, 1.0, 0.0]),
        Vertex::new([-1.0, 1.0, -1.0], [0.0, 1.0, 1.0]),
        Vertex::new([-1.0, -1.0, 1.0], [1.0, 0.0, 1.0]),
        Vertex::new([1.0, -1.0, 1.0], [1.0, 1.0, 0.0]),
        Vertex::new([1.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
        Vertex::new([-1.0, 1.0, 1.0], [1.0, 0.0, 1.0]),
    ];

    let indices = vec![
        3, 1, 0, 2, 1, 3, 2, 5, 1, 6, 5, 2, 6, 4, 5, 7, 4, 6, 7, 0, 4, 3, 0, 7, 7, 2, 3, 6, 2, 7,
        0, 5, 4, 1, 5, 0,
    ];

    (vertices, indices)
}
