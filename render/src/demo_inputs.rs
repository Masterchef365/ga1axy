use crate::RenderSettings;

pub fn demo(cfg: &RenderSettings) -> (Vec<f32>, Vec<u8>) {
    let mut points = vec![];
    let mut images = vec![];
    for _ in 0..cfg.batch_size {
        demo_points(&mut points, cfg);
        demo_images(&mut images, cfg);
    }
    (points, images)
}

pub fn demo_points(points: &mut Vec<f32>, cfg: &RenderSettings) {
    points.extend(
        (0..cfg.input_points)
            .map(|i| [i as f32, 0., 0., i as f32])
            .flatten()
    );
}

pub fn demo_images(images: &mut Vec<u8>, cfg: &RenderSettings) {
    let r = cfg.input_width as i32 / 2;
    for layer in 0..cfg.input_images {
        let mut color = [0, 0, 0, 0xFF];
        color[layer as usize % 3] = 0xFF;

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