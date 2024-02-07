import streamlit as st
from PIL import Image
from inference import get_text
from model import vit_model
from inference import feature_extractor,tokenizer


def main():
    st.title("Image to Text Generation App")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate text from the uploaded image
        generated_text = get_text(image,vit_model,tokenizer)
        st.subheader("Generated Text:")
        st.write(generated_text)

if __name__ == "__main__":
    main()
