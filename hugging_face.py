# utils/meals_generate_function.py

import json
import logging
import asyncio
from typing import Dict, List, Optional, Set
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
import base64
import json
import os
import logging
from dotenv import load_dotenv
import aiohttp
import requests
from PIL import Image
import io

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")  

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")


class ImageGenerationService:
    """Service for generating meal images using Hugging Face Stable Diffusion"""
    
    def __init__(self, huggingface_api_key: str):
        self.api_key = huggingface_api_key
        self.api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        self.headers = {"Authorization": f"Bearer {huggingface_api_key}"}
    
    async def generate_meal_image(self, meal_name: str, description: str, cuisine_type: str) -> Optional[str]:
        """Generate an image for a meal and return base64 encoded string"""
        try:
            # Create a detailed prompt for food photography
            prompt = f"""
            Professional food photography of {meal_name}, {description}, 
            {cuisine_type} cuisine, beautifully plated on elegant dishware, 
            natural lighting, high quality, appetizing, restaurant style presentation, 
            garnished, colorful, vibrant, detailed texture, shallow depth of field
            """
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        # Convert to base64
                        base64_image = base64.b64encode(image_bytes).decode('utf-8')
                        return f"data:image/png;base64,{base64_image}"
                    else:
                        logger.warning(f"Image generation failed for {meal_name}: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error generating image for {meal_name}: {str(e)}")
            return None