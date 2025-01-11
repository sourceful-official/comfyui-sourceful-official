import replicate
import torch
import io
import base64
import torchvision.transforms as transforms
import requests
from PIL import Image
import numpy as np
from fal_client import submit


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
                "prompt": ("STRING", {"default": ""}),
                "logo_image": ("IMAGE",),
                "target_images": ("IMAGE", {"array": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "comfyui_incontext_three_panels"

    def __init__(self):
        pass

    def comfyui_incontext_three_panels(self, prompt, logo_image, target_images):
        print("prompt", prompt)
        print("logo_image", logo_image)
        print("target_image", target_images)
        predictions = []
        print("target_images.shape", target_images.shape)
        for i in range(target_images.shape[0]):
            target_image = target_images[i]
            target_image = target_image.unsqueeze(0) 
            print("target_image.shape", target_image.shape)
            deployment = replicate.deployments.get(
                "sourceful-official/cog-comfyui-incontext-three-panels")
            prediction = deployment.predictions.create(
                input={
                    "prompt": prompt, 
                    "logo_image": image_to_base64(logo_image), 
                    "target_image": image_to_base64(target_image)}
            )
            predictions.append(prediction)
        for prediction in predictions:
            prediction.wait()
        for prediction in predictions:
            print("prediction.output", prediction.output)
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
            predictions.append(tensor_image)
        return (predictions,)
    
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


NODE_CLASS_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": SourcefulOfficialComfyuiIncontextThreePanels,
    "FalFluxLoraSourcefulOfficial": FalFluxLoraSourcefulOfficial,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": "SourcefulOfficialComfyuiIncontextThreePanels",
    "FalFluxLoraSourcefulOfficial": "FalFluxLoraSourcefulOfficial",
}
