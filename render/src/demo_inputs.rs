use crate::{Input, RenderSettings};
use rand::Rng;

pub fn demo(cfg: &RenderSettings) -> Input {
    let mut points = vec![];
    let mut images = vec![];
    for sample in 0..cfg.batch_size {
        demo_points(&mut points, cfg, sample);
        demo_images(&mut images, cfg);
    }
    Input { points, images }
}

pub fn demo_points(points: &mut Vec<f32>, cfg: &RenderSettings, sample: u32) {
    points.extend(
        (0..cfg.input_points)
            .map(|i| {
                let frac = i as f32 / cfg.input_points as f32;
                let mut point = [0., 0., 0., 1. - frac];
                point[sample as usize % 3] = i as f32;
                point
            })
            .flatten(),
    );
}

pub fn demo_images(images: &mut Vec<u8>, cfg: &RenderSettings) {
    let r = cfg.input_width as i32 / 2;
    for layer in 0..cfg.input_images {
        let frac = ((layer * 255) / cfg.input_images) as u8;
        let mut color = [0, 0, 0, 0xFF];
        color[layer as usize % 3] = frac;

        for y in 0..cfg.input_height {
            for x in 0..cfg.input_width {
                let x = x as i32 - (cfg.input_width as i32 / 2);
                let y = y as i32 - (cfg.input_height as i32 / 2);
                let z = layer as i32 - (cfg.input_images as i32 / 2);

                let inside = x * x + y * y + z * z < r * r;
                let pixel = if inside { color } else { [0, 0, 0, 0] };

                images.extend_from_slice(&pixel);
            }
        }
    }
}

pub fn random(cfg: &RenderSettings) -> Input {
    let mut rng = rand::thread_rng();
    let points = (0..cfg.batch_size * cfg.input_points * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let images =
        (0..cfg.batch_size * cfg.input_images * cfg.input_height * cfg.input_width * 4)
            .map(|_| rng.gen_range(u8::MIN..u8::MAX))
            .collect();
    Input { points, images }
}
