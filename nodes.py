import replicate


class SourcefulOfficialComfyuiIncontextThreePanels:
    CATEGORY = "sourceful-official"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "logo_image": ("IMAGE",),
                "target_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING[]",)
    FUNCTION = "comfyui_incontext_three_panels"

    def __init__(self):
        pass

    def comfyui_incontext_three_panels(self, prompt, logo_image, target_image):
        deployment = replicate.deployments.get(
            "sourceful-official/cog-comfyui-incontext-three-panels")
        prediction = deployment.predictions.create(
            input={"prompt": prompt, "logo_image": logo_image, "target_image": target_image}
        )
        prediction.wait()
        return (prediction.output,)

NODE_CLASS_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": SourcefulOfficialComfyuiIncontextThreePanels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SourcefulOfficialComfyuiIncontextThreePanels": "SourcefulOfficialComfyuiIncontextThreePanels",
}