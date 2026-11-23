"""Exact 720p output from 32-pixel-aligned MiniMax H3 latent dimensions."""

CROPS = {
    (1280, 736): (1280, 720, 0, 8),
    (736, 1280): (720, 1280, 8, 0),
}


def apply_720p_crop(workflow, width, height, output_resolution=None):
    """Crop decoded images before CreateVideo; keep original audio unchanged."""
    if output_resolution in (None, "native"):
        return workflow
    if output_resolution != "720p" or (width, height) not in CROPS:
        raise ValueError("720p output requires 1280x736 or 736x1280 input.")

    videos = [node for node in workflow.values()
              if node.get("class_type") == "CreateVideo"]
    if len(videos) != 1 or "images" not in videos[0].get("inputs", {}):
        raise ValueError("Expected one CreateVideo node with decoded images.")

    out_width, out_height, x, y = CROPS[(width, height)]
    crop_id = str(max((int(key) for key in workflow if str(key).isdigit()), default=0) + 1)
    original_images = videos[0]["inputs"]["images"]
    workflow[crop_id] = {
        "class_type": "ImageCrop",
        "inputs": {"image": original_images, "width": out_width,
                   "height": out_height, "x": x, "y": y},
    }
    videos[0]["inputs"]["images"] = [crop_id, 0]
    return workflow
