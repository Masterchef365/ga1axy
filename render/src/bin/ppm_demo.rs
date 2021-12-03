use anyhow::Result;
use ga1axy::RenderSettings;
use ga1axy::trainer::Trainer;
use ga1axy::ppm::write_output_as_ppm;
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
        enable_depth: true,
    };

    let mut trainer = Trainer::new(cfg)?;
    let input = ga1axy::demo_inputs::random(&cfg);
    //let input = demo(&cfg);

    let start = std::time::Instant::now();

    let output = trainer.frame(&input)?;

    let end = start.elapsed();
    println!("Frame took {}s", end.as_secs_f32());

    write_output_as_ppm("", "", &output, &cfg)?;

    Ok(())
}