use anyhow::Result;
use crate::{RenderSettings, Input, Output, engine::Engine};
use watertender::defaults::DEPTH_FORMAT;
use watertender::headless_backend::build_core;
use watertender::memory::{UsageFlags, ManagedImage, ManagedBuffer};
use watertender::app_info::AppInfo;
use watertender::{vk, SharedCore, Core};
use std::sync::Arc;
use nalgebra::{Matrix4, Point3, Vector3};
use rand::Rng;

pub struct Trainer {
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    render_pass: vk::RenderPass,

    framebuffer: vk::Framebuffer,
    fb_image: ManagedImage,
    fb_image_view: vk::ImageView,
    _fb_depth_image: ManagedImage,
    fb_depth_image_view: vk::ImageView,
    fb_download_buf: ManagedBuffer,
    fb_extent: vk::Extent2D,
    fb_size_bytes: u64,

    engine: Engine,
    core: SharedCore,
}

fn random_arcball(rng: &mut impl Rng) -> Matrix4<f32> {
    use std::f32::consts::{PI, FRAC_PI_2};
    let pitch = rng.gen_range(-FRAC_PI_2..FRAC_PI_2);
    let yaw = rng.gen_range(-PI..PI);
    let distance = rng.gen_range(2.0..8.0);
    let fov = PI / 3.;
    arcball(pitch, yaw, distance, fov)
}

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

impl Trainer {
    pub fn new(cfg: RenderSettings) -> Result<Self> {
        let info = AppInfo::default()
            .validation(cfg!(debug_assertions))
            .vk_version(1, 1, 0);
        let core = build_core(info)?;
        let core = Arc::new(core);

        // Command pool
        let create_info = vk::CommandPoolCreateInfoBuilder::new()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(core.queue_family);
        let command_pool =
            unsafe { core.device.create_command_pool(&create_info, None, None) }.result()?;

        // Allocate command buffers
        let allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer =
            unsafe { core.device.allocate_command_buffers(&allocate_info) }.result()?[0];

        // Create render pass
        let render_pass = create_render_pass(&core)?;

        // Create engine
        let engine = Engine::new(core.clone(), cfg, render_pass, command_buffer)?;

        // Create frame download staging buffer
        let fb_size_bytes = (cfg.output_height * cfg.output_width * 4) as u64 * std::mem::size_of::<u8>() as u64;
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .size(fb_size_bytes);

        let fb_download_buf = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::DOWNLOAD,
        )?;

        // Output extent
        let fb_extent = vk::Extent2DBuilder::new()
            .width(cfg.output_width)
            .height(cfg.output_height)
            .build();

        // Create depth image
        let create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(
                vk::Extent3DBuilder::new()
                    .width(fb_extent.width)
                    .height(fb_extent.height)
                    .depth(1)
                    .build(),
            )
            .mip_levels(1)
            .array_layers(1)
            .format(DEPTH_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let fb_depth_image = ManagedImage::new(
            core.clone(),
            create_info,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(fb_depth_image.instance())
            .view_type(vk::ImageViewType::_2D)
            .format(DEPTH_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        let fb_depth_image_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }.result()?;

        // Build image views and buffers
        let create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .extent(
                vk::Extent3DBuilder::new()
                    .width(fb_extent.width)
                    .height(fb_extent.height)
                    .depth(1)
                    .build(),
            )
            .mip_levels(1)
            .array_layers(1)
            .format(COLOR_FORMAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .samples(vk::SampleCountFlagBits::_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let fb_image = ManagedImage::new(
            core.clone(),
            create_info,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        let create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(fb_image.instance())
            .view_type(vk::ImageViewType::_2D)
            .format(COLOR_FORMAT)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        let fb_image_view =
            unsafe { core.device.create_image_view(&create_info, None, None) }
                .result()?;

        let attachments = [fb_image_view, fb_depth_image_view];
        let create_info = vk::FramebufferCreateInfoBuilder::new()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(fb_extent.width)
            .height(fb_extent.height)
            .layers(1);

        let framebuffer = unsafe {
            core
                .device
                .create_framebuffer(&create_info, None, None)
        }
        .result()?;

        Ok(Self {
            render_pass,
            framebuffer,
            fb_image,
            fb_image_view,
            _fb_depth_image: fb_depth_image,
            fb_depth_image_view,
            fb_extent,
            fb_size_bytes,
            fb_download_buf,
            core,
            command_buffer,
            command_pool,
            engine
        })
    }

    pub fn frame(&mut self, input: &Input) -> Result<Output> {
        let mut output = vec![];
        let cfg = self.engine.cfg();

        for frame in 0..cfg.batch_size as usize {
            output.extend(self.single_frame(frame, input)?);
        }

        Ok(Output {
            images: output
        })
    }

    /*fn single_frame(&self, point_data: &[f32], image_data: &[u8]) -> Result<Vec<u8>> {
        self.
    }*/

    fn single_frame(&mut self, idx: usize, input: &Input) -> Result<Vec<u8>> {
        self.engine.upload(idx, input)?;
        self.engine.prepare(self.command_buffer)?;

        let mut rng = rand::thread_rng();
        let camera = random_arcball(&mut rng);

        // Record command buffer to upload to gpu_buffer
        unsafe {
            self
                .core
                .device
                .reset_command_buffer(self.command_buffer, None)
                .result()?;
            let begin_info = vk::CommandBufferBeginInfoBuilder::new();
            self
                .core
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)
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
                .image(self.fb_image.instance())
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .subresource_range(image_subresource);

            self.core.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                None,
                &[],
                &[],
                &[barrier],
            );

            // Set render pass
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let begin_info = vk::RenderPassBeginInfoBuilder::new()
                .framebuffer(self.framebuffer)
                .render_pass(self.render_pass)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.fb_extent,
                })
                .clear_values(&clear_values);

            self.core.device.cmd_begin_render_pass(
                self.command_buffer,
                &begin_info,
                vk::SubpassContents::INLINE,
            );


            let viewports = [vk::ViewportBuilder::new()
                .x(0.0)
                .y(0.0)
                .width(self.fb_extent.width as f32)
                .height(self.fb_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)];

            let scissors = [vk::Rect2DBuilder::new()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(self.fb_extent)];

            self.core
                .device
                .cmd_set_viewport(self.command_buffer, 0, &viewports);

            self.core
                .device
                .cmd_set_scissor(self.command_buffer, 0, &scissors);

            let mut camera_data = [0.; 4 * 4 * 2];
            camera_data[..4*4].copy_from_slice(camera.as_slice());
            self.engine.write_commands(self.command_buffer, 0, camera_data)?;

            self.core.device.cmd_end_render_pass(self.command_buffer);

            // Barrier (UNDEFINED -> TRANSFER_DST_OPTIMAL)
            let barrier = vk::ImageMemoryBarrierBuilder::new()
                .image(self.fb_image.instance())
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .subresource_range(image_subresource);

            self.core.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                None,
                &[],
                &[],
                &[barrier],
            );

            // Image copy to download buffer
            let sub_layers = vk::ImageSubresourceLayersBuilder::new()
                .layer_count(1)
                .base_array_layer(0)
                .mip_level(0)
                .aspect_mask(vk::ImageAspectFlags::COLOR);

            let buffer_image_copy = vk::BufferImageCopyBuilder::new()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_extent(vk::Extent3DBuilder::new().width(self.fb_extent.width).height(self.fb_extent.height).depth(1).build())
                .image_offset(vk::Offset3DBuilder::new().x(0).y(0).z(0).build())
                .image_subresource(*sub_layers);

            self.core.device.cmd_copy_image_to_buffer(
                self.command_buffer,
                self.fb_image.instance(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.fb_download_buf.instance(),
                &[buffer_image_copy]
            );

            // Submit & wait
            self
                .core
                .device
                .end_command_buffer(self.command_buffer)
                .result()?;
            let command_buffers = [self.command_buffer];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&command_buffers);
            self
                .core
                .device
                .queue_submit(self.core.queue, &[submit_info], None)
                .result()?;
            self
                .core.device.queue_wait_idle(self.core.queue).result()?;

        }

        let mut image_data = vec![0xFFu8; self.fb_size_bytes as usize];
        self.fb_download_buf.read_bytes(0, &mut image_data)?;

        // Convert RGBA to RBG
        let image_data = rgba_to_rgb(image_data);

        Ok(image_data)
    }
}

fn rgba_to_rgb(input: Vec<u8>) -> Vec<u8> {
    assert!(input.len() % 4 == 0);
    let mut output = Vec::with_capacity((input.len() * 3) / 4);
    for pixel in input.chunks_exact(4) {
        output.push(pixel[0]);
        output.push(pixel[1]);
        output.push(pixel[2]);
    }
    output
}

pub fn create_render_pass(core: &Core) -> Result<vk::RenderPass> {
    let device = &core.device;

    // Render pass
    let color_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(COLOR_FORMAT)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(DEPTH_FORMAT)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let attachments = [color_attachment, depth_attachment];

    let color_attachment_refs = [vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let depth_attachment_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();

    let subpasses = [vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)];

    let dependencies = [vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let create_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    /*
    let views = 1;
    let view_mask = [!(!0 << views)];
    let mut multiview = erupt::vk1_1::RenderPassMultiviewCreateInfoBuilder::new()
        .view_masks(&view_mask)
        .correlation_masks(&view_mask)
        .build();

    create_info.p_next = &mut multiview as *mut _ as _;
    */


    Ok(unsafe { device.create_render_pass(&create_info, None, None) }.result()?)
}

impl Drop for Trainer {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_command_pool(Some(self.command_pool), None);
            self.core.device.destroy_framebuffer(Some(self.framebuffer), None);
            self.core.device.destroy_render_pass(Some(self.render_pass), None);
            self.core.device.destroy_image_view(Some(self.fb_image_view), None);
            self.core.device.destroy_image_view(Some(self.fb_depth_image_view), None);
        }
    }
}

fn arcball(pitch: f32, yaw: f32, distance: f32, fov: f32) -> Matrix4<f32> {
    let eye = Point3::new(
        yaw.cos() * pitch.cos() * distance,
        pitch.sin() * distance,
        yaw.sin() * pitch.cos() * distance,
    );

    let view = Matrix4::look_at_rh(&eye, &Point3::origin(), &Vector3::y());
    let projection = Matrix4::new_perspective(
        1.,
        fov,
        0.001,
        1000.,
    );
    projection * view
}
