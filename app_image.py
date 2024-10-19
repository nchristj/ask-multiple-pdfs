import streamlit as st
import base64
from PIL import Image
import io

def image_to_base64(image: Image.Image) -> str:
    """Convert an image to a Base64 string."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')  # Save image in PNG format
    img_bytes.seek(0)  # Go to the start of the BytesIO buffer
    base64_str = base64.b64encode(img_bytes.getvalue()).decode('utf-8')  # Convert to Base64
    return base64_str

# Set the title of the app
st.title("Image Upload and Base64 Conversion")

# File uploader for image input with drag-and-drop
uploaded_file = st.file_uploader("Drag and drop an image file here...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Calculate Base64 encoding of the image
    base64_image = image_to_base64(image)

    # Display the Base64 string (you can also save it or use it as needed)
    st.subheader("Base64 Encoded Image")
    st.text_area("Base64 String", base64_image, height=300)

# Optional: To clear the uploaded image
if st.button("Clear Image"):
    st.session_state.clear()
