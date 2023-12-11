from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import streamlit as st
import requests

# Load model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Function to load image from computer
def load_image_from_file(file):
    image = Image.open(file)
    return image

# Function to load image from URL
def load_image_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw)
    return image

# Function to predict the class
def predict(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

def main():
    st.set_page_config(layout="wide")
    st.title("ViT Image Classification App")

    col1, col2 = st.columns([2, 3])
    image = None

    # Left column: Buttons, upload fields, and Predict Class button
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        url = st.text_input("Enter image URL:")
        if st.button("Predict Class") and (url or uploaded_file):
            if uploaded_file:
                image = load_image_from_file(uploaded_file)
            elif url:
                try:
                    image = load_image_from_url(url)
                except Exception as e:
                    st.warning(f"Error loading image from URL: {e}")
                    return
            else:
                st.warning("Please upload an image or provide an image URL.")
                return

            predicted_class = predict(image)
            st.success(f"Predicted class: {predicted_class}")

    # Right column: Display the uploaded image
    with col2:
        if image:
            # To ensure that square images do not use the entire width of the column
            aspect_ratio = image.width / image.height
            use_column_width = aspect_ratio > 1.4

            st.caption("Uploaded image:")
            if uploaded_file:
                st.image(image, caption="Uploaded Image", use_column_width=use_column_width)
            elif url:
                try:
                    st.image(image, caption="Image from URL", use_column_width=use_column_width)
                except Exception as e:
                    st.warning(f"Error loading image from URL: {e}")

if __name__ == "__main__":
    main()
