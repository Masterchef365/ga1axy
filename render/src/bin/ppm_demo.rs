use anyhow::Result;
use ga1axy::RenderSettings;
use ga1axy::trainer::Trainer;
//use ga1axy::demo_inputs::demo;

fn main() -> Result<()> {
    let cfg = RenderSettings {
        batch_size: 10,
        output_width: 256,
        output_height: 256,
        input_images: 10,
        input_width: 64,
        input_height: 64,
        input_points: 128,
        background_color: [0.; 4],
    };

    let mut trainer = Trainer::new(cfg)?;
    let input = ga1axy::demo_inputs::random(&cfg);
    //let input = demo(&cfg);

    let start = std::time::Instant::now();
    let output = trainer.frame(&input)?;
    let end = start.elapsed();
    println!("Frame took {}s", end.as_secs_f32());

    for (idx, frame) in output.image_arrays(&cfg).enumerate() {
        let path = format!("{}.ppm", idx);
        save_image(path, &frame, cfg.output_width as _)?;
    }

    Ok(())
}

use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

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
