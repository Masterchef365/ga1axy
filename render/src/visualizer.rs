use watertender::prelude::*;
use anyhow::Result;
use crate::{Input, RenderSettings, engine::Engine};

struct Visualizer {
    engine: Engine,
    camera: MultiPlatformCamera,
    starter_kit: StarterKit,
}

type RenderInputs = (Input, RenderSettings);

pub fn visualize(input: Input, cfg: RenderSettings, vr: bool) -> Result<()> {
    crate::verify_input(&input, &cfg)?;
    let info = AppInfo::default().validation(cfg!(debug_assertions));
    launch::<Visualizer, RenderInputs>(info, vr, (input, cfg))
}


impl MainLoop<RenderInputs> for Visualizer {
    fn new(core: &SharedCore, mut platform: Platform<'_>, (input, cfg): RenderInputs) -> Result<Self> {
        let starter_kit = StarterKit::new(core.clone(), &mut platform)?;

        let camera = MultiPlatformCamera::new(&mut platform);
        let mut engine = Engine::new(core.clone(), cfg, starter_kit.render_pass, starter_kit.current_command_buffer())?;

        engine.upload(0, &input)?;
        engine.prepare(starter_kit.current_command_buffer())?;

        Ok(Self {
            camera,
            starter_kit,
            engine,
        })
    }

    fn frame(
        &mut self,
        frame: Frame,
        _core: &SharedCore,
        platform: Platform<'_>,
    ) -> Result<PlatformReturn> {
        let cmd = self.starter_kit.begin_command_buffer(frame)?;
        let command_buffer = cmd.command_buffer;
        self.starter_kit.begin_render_pass(command_buffer, frame, self.engine.cfg().background_color);

        let (ret, cameras) = self.camera.get_matrices(&platform)?;

        self.engine.write_commands(command_buffer, self.starter_kit.frame, cameras)?;

        self.starter_kit.end_command_buffer(cmd)?;

        Ok(ret)
    }

    fn swapchain_resize(&mut self, images: Vec<vk::Image>, extent: vk::Extent2D) -> Result<()> {
        self.starter_kit.swapchain_resize(images, extent)
    }

    fn event(
        &mut self,
        mut event: PlatformEvent<'_, '_>,
        _core: &Core,
        mut platform: Platform<'_>,
    ) -> Result<()> {
        self.camera.handle_event(&mut event, &mut platform);
        starter_kit::close_when_asked(event, platform);
        Ok(())
    }
}

impl SyncMainLoop<RenderInputs> for Visualizer {
    fn winit_sync(&self) -> (vk::Semaphore, vk::Semaphore) {
        self.starter_kit.winit_sync()
    }
}
