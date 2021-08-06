use ga1axy_render::{RenderSettings, demo_inputs::demo, visualize, Input};
use anyhow::Result;

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

    let (points, images) = demo(&cfg);

    let input = Input {
        points: &points,
        images: &images,
    };

    visualize(input, &cfg)
}
