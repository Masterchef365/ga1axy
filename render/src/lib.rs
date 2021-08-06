pub mod demo_inputs;
pub mod visualizer;
pub use visualizer::visualize;

use anyhow::{Result, ensure};

pub struct Render {
    cfg: RenderSettings,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderSettings {
    /// Batch size
    pub batch_size: u32,
    /// Width dimension of the rendered image
    pub output_width: u32,
    /// Height dimension of the rendered image
    pub output_height: u32,
    /// Number of input images
    pub input_images: u32,
    /// Width of input images
    pub input_width: u32,
    /// Height of input images
    pub input_height: u32,
    /// Number of input points
    pub input_points: u32,
}

pub struct Input<'a> {
    /// Pointcloud data, logically organized (batch_size, input_points, 4)
    /// XYZW where W is the texture index
    pub points: &'a [f32],
    /// Images data, logically organized (batch_size, input_images, input_height, input_width, 4), RGBA
    pub images: &'a [u8],
}

pub struct Output {
    /// Image data, logically organized (batch_size, output_height, output_width, 3), RGB
    pub images: Vec<u8>,
}

impl Render {
    pub fn new(cfg: RenderSettings) -> Result<Self> {
        Ok(Self {
            cfg,
        })
    }

    pub fn render(&mut self, input: Input<'_>) -> Result<Output> {
        verify_input(input, &self.cfg)?;
        todo!()
    }
}

fn verify_input(input: Input<'_>, cfg: &RenderSettings) -> Result<()> {
    let expect_points = cfg.batch_size * cfg.input_points * 4;
    ensure!(input.points.len() as u32 == expect_points, "Expected {} values for points, got {}", expect_points, input.points.len());
    let expect_images = cfg.batch_size * cfg.input_images * cfg.input_height * cfg.input_width * 4;
    ensure!(input.images.len() as u32 == expect_images, "Expected {} values for images, got {}", expect_images, input.images.len());
    Ok(())
}