use anyhow::Result;
use ga1axy::{visualize, RenderSettings};
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
    };

    //let input = demo(&cfg);
    let input = ga1axy::demo_inputs::random(&cfg);

    visualize(input, cfg, false)
}
