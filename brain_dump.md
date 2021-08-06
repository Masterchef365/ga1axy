How to handle opacity?
* I think just set a blend mode and let the net do it's thing man

How to handle the W channel (after XYZ)?
* We could use it to index into an image cube. I think we'll just do that

How to render?
* Just do quads with a fixed scale, and use instancing

What model to use?
* https://github.com/unit8co/vegans
    * Use VanillaGAN and our own `generator`

What data to use?
* https://astronn.readthedocs.io/en/latest/galaxy10.html

# Conversation with the guy in the Vulkan Discord
[11:32 PM] Seg: I want to render maybe 100 independent and very simple scenes (like single draw calls) to very small images (like 256x256) - I would also like to do so in a very short amount of time. How can I avoid using a full "frame" to render each image? My first idea was to use multiview and a layered framebuffer, but I realized that 100 is too many for multiview and my scenes are all distinct anyway. My next idea was to create a single framebuffer large enough to accomodate all of the samples, and then use dynamic scissor and viewport to switch where I'm drawing for each draw call. Would this be slow and/or impossible? I'm not sure where to look in the spec to find out whether or not I can set the viewport or scissor in between draw calls.
[11:42 PM] pixelcluster: Have you already tried simply rendering each scene to a separate image? If yes and it was too slow, you can indeed set the viewport/scissor after each draw if the respective dynamic states are activated. If you haven’t tried the naive approach, try that first and see if it’s already enough
[11:45 PM] Seg: Indeed I haven't tried the slow idea yet, and I realize that premature optimization is a great way to arrive at overcomplicated code - I'll definitely try the slow way first, but I'm trying to get ahead of this problem because it's to generate datasets and "fast enough" is really "as fast as I can possibly make it"
thanks for the advice though, I'll try setting scissor/viewport per draw call if I find that it's not enough data
