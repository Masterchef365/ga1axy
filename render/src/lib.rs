pub mod demo_inputs;
pub mod visualizer;
mod engine;
pub mod trainer;
pub use visualizer::visualize;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray5, PyReadonlyArray3, PyArray4, PyArray};
pub mod ppm;

use anyhow::{ensure, Result};

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
    /// Background color
    pub background_color: [f32; 4],
    /// Enable depth test
    pub enable_depth: bool,
}

#[derive(Clone)]
pub struct Input {
    /// Pointcloud data, logically organized (batch_size, input_points, 4)
    /// XYZW where W is the texture index
    pub points: Vec<f32>,
    /// Images data, logically organized (batch_size, input_images, input_height, input_width, 4), RGBA
    pub images: Vec<u8>,
    /// Camera matrices, organized into (batch_size, columns, rows)
    pub cameras: Vec<f32>,
}

pub struct Output {
    /// Image data, logically organized (batch_size, output_height, output_width, 3), RGB
    pub images: Vec<u8>,
}

impl Output {
    /// Returns array slices corresponding to the images in this output
    pub fn image_arrays(&self, cfg: &RenderSettings) -> impl Iterator<Item=&[u8]> {
        let frame_len = cfg.output_width * cfg.output_height * 3;
        self.images.chunks_exact(frame_len as _)
    }
}

impl Render {
    pub fn new(cfg: RenderSettings) -> Result<Self> {
        Ok(Self { cfg })
    }

    pub fn render(&mut self, input: Input) -> Result<Output> {
        verify_input(&input, &self.cfg)?;
        todo!()
    }
}

fn verify_input(input: &Input, cfg: &RenderSettings) -> Result<()> {
    ensure!(
        input.points.len() as u32 == points_float_count(&cfg),
        "Expected {} values for points, got {}",
        points_float_count(&cfg),
        input.points.len()
    );

    ensure!(
        input.images.len() as u32 == images_byte_count(&cfg),
        "Expected {} values for images, got {}",
        images_byte_count(&cfg),
        input.images.len()
    );

    Ok(())
}

pub fn points_float_count(cfg: &RenderSettings) -> u32 {
    cfg.batch_size * cfg.input_points * 4
}

pub fn images_byte_count(cfg: &RenderSettings) -> u32 {
    cfg.batch_size * cfg.input_images * cfg.input_height * cfg.input_width * 4
}

#[pymodule]
fn ga1axy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTrainer>()?;
    m.add_wrapped(wrap_pyfunction!(visualize_inputs))?;
    Ok(())
}

#[pyclass]
struct PyTrainer {
    trainer: trainer::Trainer,
    cfg: RenderSettings,
}

#[pymethods]
impl PyTrainer {
    #[new]
    pub fn new(
        batch_size: u32,
        output_width: u32,
        output_height: u32,
        input_images: u32,
        input_width: u32,
        input_height: u32,
        input_points: u32,
        background_color: [f32; 4],
        enable_depth: bool,
    ) -> PyResult<Self> {
        let cfg = RenderSettings {
            batch_size,
            output_width,
            output_height,
            input_images,
            input_width,
            input_height,
            input_points,
            background_color,
            enable_depth,
        };

        let trainer = trainer::Trainer::new(cfg).map_err(to_py_excp)?;
        Ok(Self {
            cfg,
            trainer
        })
    }

    fn frame(&mut self, py: Python, points: PyReadonlyArray3<f32>, images: PyReadonlyArray5<u8>, cameras: PyReadonlyArray3<f32>) -> PyResult<Py<PyArray4<u8>>> {
        let input = construct_input(points, images, cameras)?;
        let output = self.trainer.frame(&input).map_err(to_py_excp)?;

        let images = PyArray::from_vec(py, output.images);
        let images = images.reshape((
            self.cfg.batch_size as usize,
            self.cfg.output_height as usize,
            self.cfg.output_width as usize,
            3,
        ))?;

        Ok(images.to_owned())
    }
}

fn construct_input(points: PyReadonlyArray3<f32>, images: PyReadonlyArray5<u8>, cameras: PyReadonlyArray3<f32>) -> PyResult<Input> {
    Ok(Input {
        points: points.as_slice()?.to_vec(),
        images: images.as_slice()?.to_vec(),
        cameras: cameras.as_slice()?.to_vec(),
    })
}

#[pyfunction]
pub fn visualize_inputs(points: PyReadonlyArray3<f32>, images: PyReadonlyArray5<u8>, cameras: PyReadonlyArray3<f32>, background_color: [f32; 4]) -> PyResult<()> {
    let batch_size = points.shape()[0] as u32;
    let input_points = points.shape()[1] as u32;
    let point_channels = points.shape()[2] as u32;

    let batch_size_img = images.shape()[0] as u32;
    let input_images = images.shape()[1] as u32;
    let input_height = images.shape()[2] as u32;
    let input_width = images.shape()[3] as u32;
    let img_channels = images.shape()[4] as u32;

    let input = construct_input(points, images, cameras)?;

    assert_eq!(batch_size, batch_size_img);
    assert_eq!(point_channels, 4);
    assert_eq!(img_channels, 4);

    let cfg = RenderSettings {
        batch_size,
        input_images,
        input_width,
        input_height,
        input_points,
        output_width: 256,
        output_height: 256,
        background_color,
        enable_depth: true,
    };

    Ok(visualize(input, cfg, false).map_err(to_py_excp)?)
}


fn to_py_excp<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())
}
