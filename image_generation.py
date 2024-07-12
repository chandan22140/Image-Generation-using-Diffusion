# -*- coding: utf-8 -*-

import os
from huggingface_hub import login
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import cv2
from glob import glob

def main():
    # Authenticate with Hugging Face
    login()

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        revision="fp16", 
        torch_dtype=torch.float16, 
        use_auth_token=True
    )
    pipe = pipe.to("cuda")

    # Create directory to save images
    output_dir = "data/saved_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    prompt = "cat and dogs"
    with autocast("cuda"):
        output = pipe(prompt, height=512, width=768, num_inference_steps=50)

    print("Output keys:", output.keys())

    # Save the generated image
    for i in range(len(output["images"])):
        image = output["images"][i]
        image_path = os.path.join(output_dir, f"sample_{i}.png")
        image.save(image_path)

    # Resize images
    for img_path in glob(os.path.join(output_dir, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (300, 300))
        cv2.imwrite(img_path, img)


if __name__ == "__main__":
    main()
