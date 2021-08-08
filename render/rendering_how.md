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


In fine detail:
* Upload image data to the staging buffer
* Upload instance data directly
* Transition image from undefined to `TRANSFER_DST_OPTIMAL`
* Copy staging buffer bytes to image
* Transition image from `TRANSFER_DST_OPTIMAL` to `SHADER_READ_ONLY_OPTIMAL` (with access masks for sampling)
* 
