import replicate
import torch
import io
import base64
import torchvision.transforms as transforms
import requests
from PIL import Image
import numpy as np
import os
import tempfile
from fal_client import submit, upload_file

def upload_image(image):
    try:
        # Convert the image tensor to a numpy array
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        # Ensure the image is in the correct format (H, W, C)
        if image_np.ndim == 4:
            image_np = image_np.squeeze(0)  # Remove batch dimension if present
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
        elif image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)

        # Normalize the image data to 0-255 range
        if image_np.dtype == np.float32 or image_np.dtype == np.float64:
            image_np = (image_np * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_np)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            pil_image.save(temp_file, format="PNG")
            temp_file_path = temp_file.name

        # Upload the temporary file
        image_url = upload_file(temp_file_path)
        return image_url
    except Exception as e:
        print(f"Error uploading image: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

def image_to_base64(image):
    if isinstance(image, torch.Tensor):
        image = image.permute(0, 3, 1, 2).squeeze(0)
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)
    else:
        pil_image = image

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


class SourcefulOfficialComfyuiIncontextThreePanels:
    CATEGORY = "sourceful-official"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "" , "multiline": True}),
                "logo_image": ("IMAGE",),
                "target_images": ("IMAGE", {"array": True}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "comfyui_incontext_three_panels"

    def __init__(self):
        pass

    def comfyui_incontext_three_panels(
            self, 
            prompt, 
            logo_image, 
            target_images,
            seed=-1,
        ):
        if logo_image is None:
            return (None,)
        if target_images is None:
            return (None,)
            
        print("prompt", prompt)
        print("logo_image", logo_image)
        print("target_image", target_images)
        predictions = []
        print("target_images.shape", target_images.shape)
        for i in range(target_images.shape[0]):
            target_image = target_images[i]
            target_image = target_image.unsqueeze(0) 
            print("target_image.shape", target_image.shape)

            input = {
                "prompt": prompt, 
                "logo_image": image_to_base64(logo_image), 
                "target_image": image_to_base64(target_image)
            }
            if seed != -1:
                input["seed"] = seed
            deployment = replicate.deployments.get(
                "sourceful-official/cog-comfyui-incontext-three-panels")
            prediction = deployment.predictions.create(
                input=input
            )
            predictions.append(prediction)
        for prediction in predictions:
            prediction.wait()
        results = []
        for prediction in predictions:
            print("prediction.output", prediction)
            output_url = prediction.output[1]
            print("output_url", output_url)
            transform = transforms.ToTensor()
            response = requests.get(output_url)
            image = Image.open(io.BytesIO(response.content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor_image = transform(image)
            tensor_image = tensor_image.unsqueeze(0)
            tensor_image = tensor_image.permute(0, 2, 3, 1).cpu().float()
            results.append(tensor_image)
        final_tensor = torch.cat(results, dim=0)
        return (final_tensor,)
    
class FalFluxLoraSourcefulOfficial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image_size": (["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"], {"default": "landscape_4_3"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "seed": ("INT", {"default": -1}),
                "lora_path_1": ("STRING", {"default": ""}),
                "lora_scale_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "lora_path_2": ("STRING", {"default": ""}),
                "lora_scale_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "generate_image"
    CATEGORY = "sourceful-official"

    def generate_image(self, prompt, image_size, num_images, seed=-1, lora_path_1="", lora_scale_1=1.0, lora_path_2="", lora_scale_2=1.0):
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
        }

        if seed != -1:
            arguments["seed"] = seed

        # Add LoRAs
        loras = []
        if lora_path_1:
            loras.append({"path": lora_path_1, "scale": lora_scale_1})
        if lora_path_2:
            loras.append({"path": lora_path_2, "scale": lora_scale_2})
        if loras:
            arguments["loras"] = loras

        try:
            handler = submit("fal-ai/flux-lora", arguments=arguments)
            result = handler.get()
            return self.process_result(result)
        except Exception as e:
            print(f"Error generating image with FluxLora: {str(e)}")
            return self.create_blank_image()

    def process_result(self, result):
        images = []
        for img_info in result["images"]:
            img_url = img_info["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)

        # Stack the images along a new first dimension
        stacked_images = np.stack(images, axis=0)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(stacked_images)
        
        return (img_tensor,)

    def create_blank_image(self):   
        blank_img = Image.new('RGB', (512, 512), color='black')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)


class FalIcLightV2SourcefulOfficial:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"array": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "output_format": ("STRING", {"default": "jpeg", "options": ["jpeg", "png"]}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "initial_latent": ("STRING", {"default": "None", "options": ['None', 'Left', 'Right', 'Top', 'Bottom']}),
                "lowres_denoise": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.05}),
                "highres_denoise": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "background_threshold": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "generate_image"
    CATEGORY = "sourceful-official"

    def generate_image(
            self, 
            images, 
            cfg=1.0, 
            prompt="", 
            hr_downscale=0.5, 
            output_format="jpeg", 
            guidance_scale=5.0, 
            initial_latent="None",
            lowres_denoise=0.98, 
            highres_denoise=0.95, 
            num_inference_steps=28, 
            background_threshold=0.67, 
            enable_safety_checker=True,
            seed=-1,   
        ):
        handlers = []
        results = []
            
        print("hr_downscale", hr_downscale)
        for i in range(images.shape[0]):
            image = images[i]
            image = image.unsqueeze(0) 
            image_url = upload_image(image)
            arguments = {
                "cfg": cfg,
                "prompt": prompt,
                "image_url": image_url,
                "num_images": 1,
                "hr_downscale": hr_downscale,
                "output_format": output_format,
                "guidance_scale": guidance_scale,
                "initial_latent": initial_latent,
                "lowres_denoise": lowres_denoise,
                "highres_denoise": highres_denoise,
                "num_inference_steps": num_inference_steps,
                "background_threshold": background_threshold,
                "enable_safety_checker": enable_safety_checker,
            }

            if seed != -1:
                arguments["seed"] = seed
            print("arguments", arguments)
            handler = submit(
                "fal-ai/iclight-v2", 
                arguments=arguments,
            )
            handlers.append(handler)
        for handler in handlers:
            result = handler.get()
            results.append(self.process_result(result)[0])
        print("results", results)
        final_tensor = torch.cat(results, dim=0)
        return (final_tensor,)


    def process_result(self, result):
        images = []
        for img_info in result["images"]:
            img_url = img_info["url"]
            img_response = requests.get(img_url)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)

        # Stack the images along a new first dimension
        stacked_images = np.stack(images, axis=0)
        
        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(stacked_images)
        
        return (img_tensor,)

    def create_blank_image(self):   
        blank_img = Image.new('RGB', (512, 512), color='black')
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]
        return (img_tensor,)



NODE_CLASS_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": SourcefulOfficialComfyuiIncontextThreePanels,
    "FalFluxLoraSourcefulOfficial": FalFluxLoraSourcefulOfficial,
    "FalIcLightV2SourcefulOfficial": FalIcLightV2SourcefulOfficial,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": "SourcefulOfficialComfyuiIncontextThreePanels",
    "FalFluxLoraSourcefulOfficial": "FalFluxLoraSourcefulOfficial",
    "FalIcLightV2SourcefulOfficial": "FalIcLightV2SourcefulOfficial",
}
