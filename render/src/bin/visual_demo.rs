use anyhow::Result;
use ga1axy_render::{demo_inputs::demo, visualize, RenderSettings};

fn main() -> Result<()> {
    let cfg = RenderSettings {
        batch_size: 10,
        output_width: 256,
        output_height: 256,
        input_images: 4,
        input_width: 64,
        input_height: 64,
        input_points: 128,
    };

    let input = demo(&cfg);

    visualize(input, cfg, false)
}
