use crate::{Input, RenderSettings};
use nalgebra::{Matrix4, Point3, Vector3};
use rand::Rng;

pub fn demo(cfg: &RenderSettings) -> Input {
    let mut points = vec![];
    let mut images = vec![];
    for sample in 0..cfg.batch_size {
        demo_points(&mut points, cfg, sample);
        demo_images(&mut images, cfg);
    }
    let mut rng = rand::thread_rng();
    let cameras = random_camera_data(&mut rng, cfg);
    Input { points, images, cameras }
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
    let cameras = random_camera_data(&mut rng, cfg);
    Input { points, images, cameras }
}

pub fn random_camera_data(rng: &mut impl Rng, cfg: &RenderSettings) -> Vec<f32> {
    let mut camera_data = vec![];
    for sample in 0..cfg.batch_size {
        camera_data.extend(&random_arcball(rng))
    }
    camera_data
}

pub fn random_arcball(rng: &mut impl Rng) -> Matrix4<f32> {
    use std::f32::consts::{PI, FRAC_PI_2};
    let pitch = rng.gen_range(-FRAC_PI_2..FRAC_PI_2);
    let yaw = rng.gen_range(-PI..PI);
    let distance = rng.gen_range(2.0..8.0);
    let fov = PI / 3.;
    arcball(pitch, yaw, distance, fov)
}

pub fn arcball(pitch: f32, yaw: f32, distance: f32, fov: f32) -> Matrix4<f32> {
    let eye = Point3::new(
        yaw.cos() * pitch.cos() * distance,
        pitch.sin() * distance,
        yaw.sin() * pitch.cos() * distance,
    );

    let view = Matrix4::look_at_rh(&eye, &Point3::origin(), &Vector3::y());
    let projection = Matrix4::new_perspective(
        1.,
        fov,
        0.001,
        1000.,
    );
    projection * view
}