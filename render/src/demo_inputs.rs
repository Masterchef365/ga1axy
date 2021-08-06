use crate::RenderSettings;

pub fn demo(cfg: &RenderSettings) -> (Vec<f32>, Vec<u8>) {
    let mut points = vec![];
    let mut images = vec![];
    for batch in 0..cfg.batch_size {
        demo_points_batch(&mut points, cfg);
        demo_images_batch(&mut images, cfg);
    }
    (points, images)
}

pub fn demo_points_batch(&mut Vec<f32>, cfg: &RenderSettings) {
    for i in 0..cfg.input_points {

    }
}

pub fn demo_images_batch(&mut Vec<f32>, cfg: &RenderSettings) {
    cfg.input_points
}
