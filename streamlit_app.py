import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import subprocess

# Run setup.sh to install necessary packages
subprocess.call(['./setup.sh'])

# Set page configuration
st.set_page_config(page_title="Image Filter Tool", layout="centered", initial_sidebar_state="collapsed")

# CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        h1, h2, h3, h4, h5 {
            color: #444;
        }
        .filter-button {
            font-size: 18px;
            padding: 10px 20px;
            border: none;
            background-color: #007bff;
            color: #fff;
            border-radius: 5px;
            margin: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .filter-button:hover {
            background-color: #0056b3;
        }
        img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .dark-mode {
            background-color: #1e1e2d !important;
        }
        .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5, .dark-mode p, .dark-mode label {
            color: #e1e1e1 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Image processing functions
def apply_watercolor(inp_img):
    return cv2.stylization(inp_img, sigma_s=150, sigma_r=0.25)

def apply_pencil(inp_img):
    return cv2.pencilSketch(inp_img, sigma_s=50, sigma_r=0.07, shade_factor=0.1)[0]

def apply_cartoon(inp_img):
    gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(inp_img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_sepia(inp_img):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    return cv2.transform(inp_img, sepia_filter).clip(0, 255).astype(np.uint8)

def apply_hdr(inp_img):
    lab = cv2.cvtColor(inp_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def apply_sharpen(inp_img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(inp_img, -1, kernel)

# Main function
def main():
    st.title("🎨 Stunning Image Filter Tool")
    st.write("Transform your images with artistic filters.")

    # Sidebar for theme
    dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
    if dark_mode:
        st.markdown('<style>.stApp { background-color: #1e1e2d; }</style>', unsafe_allow_html=True)

    # File uploader
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    # Filter selection
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        img_np = np.array(img)

        # Display original image
        st.subheader("Original and Processed Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_column_width=True)

        # Filters
        filters = {
            "Watercolor": apply_watercolor,
            "Pencil Sketch": apply_pencil,
            "Cartoon": apply_cartoon,
            "Sepia": apply_sepia,
            "HDR": apply_hdr,
            "Sharpen": apply_sharpen,
        }

        selected_filter = st.selectbox("Choose a filter to apply", list(filters.keys()))

        if st.button("Apply Filter"):
            with st.spinner("Applying filter..."):
                processed_img = filters[selected_filter](img_np)
                with col2:
                    st.image(processed_img, caption=f"{selected_filter} Effect", use_column_width=True)

                # Download filtered image
                buf = BytesIO()
                Image.fromarray(processed_img).save(buf, format="JPEG")
                st.download_button(
                    "Download Processed Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_filter.lower()}_effect.jpg",
                    mime="image/jpeg"
                )

    # Feedback section
    st.sidebar.subheader("Feedback")
    feedback = st.sidebar.text_area("What do you think?")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
