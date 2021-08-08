use anyhow::Result;
use crate::{RenderSettings, Input, Output, engine::Engine};
use watertender::defaults::{DEPTH_FORMAT, COLOR_FORMAT};
use watertender::headless_backend::build_core;
use watertender::prelude::*;
use watertender::memory::UsageFlags;
use std::sync::Arc;

pub struct Trainer {
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    render_pass: vk::RenderPass,

    framebuffer: vk::Framebuffer,
    fb_image: ManagedImage,
    fb_image_view: vk::ImageView,
    fb_depth_image: ManagedImage,
    fb_depth_image_view: vk::ImageView,
    fb_extent: vk::Extent2D,

    engine: Engine,
    core: SharedCore,
}

const IDENTITY_MATRICES: [f32; 4 * 4 * 2] = [
    1., 0., 0., 0., 
    0., 1., 0., 0., 
    0., 0., 1., 0., 
    0., 0., 0., 1., 
    //
    1., 0., 0., 0., 
    0., 1., 0., 0., 
    0., 0., 1., 0., 
    0., 0., 0., 1., 
];

impl Trainer {
    pub fn new(cfg: RenderSettings) -> Result<Self> {
        let info = AppInfo::default();
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

        // Create engine
        let engine = Engine::new(core.clone(), cfg, false, command_buffer)?;

        // Create render pass
        let render_pass = create_render_pass(&core, false)?;

        // Create frame download staging buffer
        let fb_size_bytes = (cfg.output_height * cfg.output_width * 4) as usize * std::mem::size_of::<f32>();
        let bi = vk::BufferCreateInfoBuilder::new()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size(fb_size_bytes as _);

        let mut fb_download_buf = ManagedBuffer::new(
            core.clone(),
            bi,
            UsageFlags::UPLOAD,
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
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
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
            fb_depth_image,
            fb_depth_image_view,
            fb_extent,
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

            // TODO: Transition framebuffer to shader write

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


            self.engine.write_commands(self.command_buffer, 0, IDENTITY_MATRICES)?;

            // TODO: Transition framebuffer to transfer dst

            // TODO: Copy the image to the download buffer

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

        todo!()
    }
}
