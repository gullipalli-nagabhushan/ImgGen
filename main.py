import streamlit as st
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(page_title="AI Image Generator", layout="centered")

# Title
st.title("🧠 AI Image Generator with Model Selector")

# Sidebar for model selection
model_option = st.selectbox(
    "Choose a Model",
    ["Stable Diffusion v1.5", "Stable Diffusion XL (SDXL)"]
)

# Prompt input
prompt = st.text_input("Enter your prompt", "A fantasy castle in the clouds")

# Load model dynamically
@st.cache_resource(show_spinner=True)
def load_model(model_name):
    if model_name == "Stable Diffusion v1.5":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
    elif model_name == "Stable Diffusion XL (SDXL)":
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        )
    else:
        raise ValueError("Invalid model name")

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        pipe = load_model(model_option)
        result = pipe(prompt)
        image = result.images[0]

        # Show the image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png"
        )
