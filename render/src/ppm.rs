use anyhow::Result;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;
use crate::{RenderSettings, Output};

pub fn write_output_as_ppm(path: impl AsRef<Path>, prefix: &str, data: &Output, cfg: &RenderSettings) -> Result<()> {
    for (idx, frame) in data.image_arrays(cfg).enumerate() {
        let path = format!("{}{}.ppm", prefix, idx);
        save_image(path, &frame, cfg.output_width as _)?;
    }
    Ok(())
}

pub fn write_ppm<W: Write>(writer: &mut W, image: &[u8], width: usize) -> Result<()> {
    let stride = width * 3;
    let height = image.len() / stride;
    debug_assert_eq!(image.len() % stride, 0);
    debug_assert_eq!(image.len() % 3, 0);
    writer.write(format!("P6\n{} {}\n255\n", width, height).as_bytes())?;
    writer.write(image)?;
    Ok(())
}

pub fn save_image<P: AsRef<Path>>(path: P, image: &[u8], width: usize) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    write_ppm(&mut file, image, width)?;
    Ok(())
}
