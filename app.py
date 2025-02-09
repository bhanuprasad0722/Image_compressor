import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import io

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Compressor", "About"])

# Home Page Content
if page == "Home":
    st.title("Welcome to PCA Image Compressor!")
    st.write("""
    This tool compresses images using **Principal Component Analysis (PCA)**.
    
    🔹 **How does it work?**  
    - PCA reduces the dimensionality of image data while keeping important details.
    - You can control the compression level using the slider.
    
    📌 **Why use this?**  
    - Reduce file sizes for faster sharing & storage.
    - Maintain good quality while saving space.
    
    Click on **Compressor** in the sidebar to start compressing images!
    """)

# Compressor Page Content
elif page == "Compressor":
    st.title("PCA Image Compressor")

    def compress_image(image, quality):
        img_array = np.array(image) / 255.0  # Normalize pixel values

        def compress_image_rgb(img_array, n_components):
            channels = []
            for i in range(3):  
                channel = img_array[:, :, i]
                pca = PCA(n_components=n_components)
                compressed = pca.fit_transform(channel)
                reconstructed = pca.inverse_transform(compressed)
                channels.append(reconstructed)
            return np.clip(np.stack(channels, axis=2), 0, 1)

        n_components = int(min(img_array.shape[:2]) * (quality / 100))
        compressed_img = compress_image_rgb(img_array, n_components)

        compressed_img = (compressed_img * 255).astype(np.uint8)
        compressed_pil = Image.fromarray(compressed_img)

        img_byte_arr = io.BytesIO()
        compressed_pil.save(img_byte_arr, format='JPEG', quality=quality)
        img_byte_arr.seek(0)

        return img_byte_arr, compressed_pil

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    quality = st.slider("Compression Level", min_value=1, max_value=100, value=50)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        compressed_img_bytes, compressed_img = compress_image(image, quality)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"Original Image ({round(len(uploaded_file.getvalue())/1024, 2)} KB)", use_container_width=True)
        with col2:
            st.image(compressed_img, caption=f"Compressed Image ({round(len(compressed_img_bytes.getvalue())/1024, 2)} KB)", use_container_width=True)

        st.download_button(
            label="Download Compressed Image",
            data=compressed_img_bytes,
            file_name="compressed_image.jpg",
            mime="image/jpeg"
        )

# About Page Content
elif page == "About":
    st.title("About This Tool")
    st.write("""
    This **PCA Image Compressor** is built using:
    
    -  **Python**  
    -  **Streamlit** (for the UI)  
    -  **PIL (Pillow)** (for image processing)  
    -  **Scikit-learn (PCA)** (for compression)  

     **Developed by:** Bhanuprasad Chellapuram 
    """)

