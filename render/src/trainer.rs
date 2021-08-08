use anyhow::Result;
use crate::{RenderSettings, Input, Output, engine::Engine};
use watertender::headless_backend::build_core;
use watertender::prelude::*;
use watertender::memory::UsageFlags;
use std::sync::Arc;

pub struct Trainer {
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    framebuffer: vk::Framebuffer,
    fb_image_view: vk::ImageView,
    fb_image: ManagedImage,
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



        Ok(Self {
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
