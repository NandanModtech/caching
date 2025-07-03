import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from torchvision import transforms
from PIL import Image
import base64
import io
import runpod
import gc


# Global model variables
transformer = None
pipe = None

def load_models():
    """Load models with proper error handling and caching"""
    global transformer, pipe
    
    if transformer is not None and pipe is not None:
        return transformer, pipe
    
    print("🔄 Loading models from cache...")
    
    try:
        transformer = FluxTransformer2DModel.from_pretrained(
            "/runpod-volume/local_catvton",
            torch_dtype=torch.bfloat16,
            local_files_only=True  # Use cached files only
        )

        pipe = FluxFillPipeline.from_pretrained(
            "/runpod-volume/local_flux",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True  # Use cached files only
        ).to("cuda")

        pipe.transformer.to(torch.bfloat16)
        
        print("✅ Models loaded successfully from cache")
        return transformer, pipe
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        # Fallback to download if cache fails
        print("🔄 Falling back to download...")
        
        transformer = FluxTransformer2DModel.from_pretrained(
            "/runpod-volume/local_catvton",
            torch_dtype=torch.bfloat16,
        )

        pipe = FluxFillPipeline.from_pretrained(
            "/runpod-volume/local_flux",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        pipe.transformer.to(torch.bfloat16)
        return transformer, pipe

# --- Utility Functions ---
def decode_base64_to_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

def encode_image_to_base64(img: Image.Image):
    """Encode PIL Image to base64 string"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_inference(image, mask, garment, prompt1 ='Detailed product shot of a clothing', prompt2='The same cloth is worn by a model in a lifestyle setting', size=(480, 640), num_steps=25, guidance_scale=30, seed=42):
    """Run the virtual try-on inference"""
    global pipe
    
    # Ensure models are loaded
    if pipe is None:
        load_models()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mask_transform = transforms.ToTensor()

    # Resize inputs
    image = image.resize(size)
    mask = mask.resize(size)
    garment = garment.resize(size)

    # Convert to tensor
    image_tensor = transform(image)
    mask_tensor = mask_transform(mask)[:1]
    garment_tensor = transform(garment)

    # Prepare composite inputs
    inpaint_image = torch.cat([garment_tensor, image_tensor], dim=2)
    garment_mask = torch.zeros_like(mask_tensor)
    extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)

    prompt = f"The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; " \
         f"[IMAGE1] {prompt1}" \
         f"[IMAGE2] {prompt2}"

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run pipeline
    with torch.no_grad():
        result = pipe(
            height=size[1],
            width=size[0] * 2,
            image=inpaint_image,
            mask_image=extended_mask,
            num_inference_steps=num_steps,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=guidance_scale,
            prompt=prompt,
        ).images[0]

    # Split result image
    width = size[0]
    garment_result = result.crop((0, 0, width, size[1]))
    tryon_result = result.crop((width, 0, width * 2, size[1]))

    # Clean up GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    return garment_result, tryon_result

def handler(event):
    """RunPod serverless handler"""
    try:
        print(f"📥 Received event")
        
        # Validate input
        if "input" not in event:
            return {"error": "Missing 'input' in event"}
        
        input_data = event["input"]
        required_fields = ["humanImage", "clothImage", "maskImage"]
        
        for field in required_fields:
            if field not in input_data:
                return {"error": f"Missing required field: {field}"}

        # Extract inputs
        human_b64 = input_data["humanImage"]
        cloth_b64 = input_data["clothImage"]
        mask_b64 = input_data["maskImage"]
        prompt1 = input_data["prompt1"]
        prompt2 = input_data["prompt2"]


        print("🔄 Decoding images...")
        
        # Decode base64 to images
        human_img = decode_base64_to_image(human_b64)
        cloth_img = decode_base64_to_image(cloth_b64)
        mask_img = decode_base64_to_image(mask_b64)

        print("🔄 Running inference...")
        
        # Run virtual try-on
        garment_img, tryon_img = run_inference(human_img, mask_img, cloth_img,prompt1,prompt2)

        print("✅ Inference completed successfully")

        # Encode results to base64
        return {
            "garment_result_base64": encode_image_to_base64(garment_img),
            "tryon_result_base64": encode_image_to_base64(tryon_img),
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Error in handler: {e}")
        return {
            "error": str(e),
            "status": "failed"
        }

# Initialize models on import (for serverless)
print("🚀 Initializing models...")
try:
    load_models()
    print("✅ Models initialized successfully")
except Exception as e:
    print(f"⚠️ Model initialization failed: {e}")

# RunPod entry point
if __name__ == "__main__":
    print("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})