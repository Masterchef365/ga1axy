use anyhow::Result;
use crate::{RenderSettings, Input, Output, engine::Engine};
use watertender::defaults::DEPTH_FORMAT;
use watertender::headless_backend::build_core;
use watertender::memory::{UsageFlags, ManagedImage, ManagedBuffer};
use watertender::app_info::AppInfo;
use watertender::{vk, SharedCore, Core};
use std::sync::Arc;

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

const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

/*const IDENTITY_MATRICES: [f32; 4 * 4 * 2] = [
    1., 0., 0., 0., 
    0., 1., 0., 0., 
    0., 0., 1., 0., 
    0., 0., 0., 1., 
    //
    1., 0., 0., 0., 
    0., 1., 0., 0., 
    0., 0., 1., 0., 
    0., 0., 0., 1., 
];*/

impl Trainer {
    pub fn new(cfg: RenderSettings) -> Result<Self> {
        let info = AppInfo::default()
            .validation(true)
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
        /*
        let point_chunk = cfg.input_points * 4;
        let image_chunk = cfg.input_images * cfg.input_height * cfg.input_width * 4;

        for (points, images) in 
            input.points.chunks_exact(point_chunk as _)
            .zip(input.images.chunks_exact(image_chunk as _)) 
        {
            output.extend(self.single_frame(points, images)?);
        }
        */

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

            self.engine.write_commands(self.command_buffer, 0, SOME_VIEW)?;

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
        let image_data = image_data.chunks_exact(4).map(|c| [c[0], c[1], c[2]]).flatten().collect();

        Ok(image_data)
    }
}

/*
const SOME_VIEW: [f32; 4 * 4 * 2] = [
    1.0309412,
    1.1592057,
    -0.51418984,
    -0.5141384,
    0.0,
    -1.7644163,
    -0.6826116,
    -0.68254334,
    -1.0204597,
    1.1711124,
    -0.51947135,
    -0.5194194,
    -0.00000069168965,
    0.0000011511867,
    17.794508,
    17.99272,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];
*/

const SOME_VIEW: [f32; 4 * 4 * 2] = [
    0.93182516,
    0.60064536,
    -0.7249493,
    -0.7248768,
    0.0,
    -2.283458,
    -0.32466766,
    -0.3246352,
    -1.1117011,
    0.5034595,
    -0.60765076,
    -0.60758996,
    -60.44499,
    -43.071396,
    81.95681,
    82.148605,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
];

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
