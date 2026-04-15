import os

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    os.path.join(TF_MODEL_ZOO_PATH, "Qwen-Image-Edit-2509"), torch_dtype=torch.bfloat16
)
print("pipeline loaded")
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image_path = f"{os.path.dirname(__file__)}/../data/edit2511input.png"
image1 = Image.open(image_path)
prompt = "这个女生看着面前的电视屏幕，屏幕上面写着“阿里巴巴”"
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_plus.png")
print("image saved at", os.path.abspath("output_image_edit_plus.png"))
