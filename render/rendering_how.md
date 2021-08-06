Init:
* Build framebuffers (`batch_size` of these)
* Build ubo (upload)
* Build instance buffer (upload)
* Build images (upload)
* Build sampler
Frame:
* Upload images
* Upload instances
* Draw each renderpass
    * Draw instances associated with this batch index
* Download framebuffers
