import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="Edge Detection R&D Project",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
    }
    h1 {
        color: #c9d1d9;
        font-weight: 600;
    }
    h2, h3 {
        color: #8b949e;
    }
    .stDownloadButton button {
        background-color: #238636;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_canny(image, threshold1, threshold2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges

def apply_sobel(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel_max = sobel.max()
    if sobel_max > 0:
        sobel = np.uint8(sobel / sobel_max * 255)
    else:
        sobel = np.uint8(sobel)
    return sobel

def apply_laplacian(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian

def apply_prewitt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    prewitt = np.sqrt(prewittx**2 + prewitty**2)
    prewitt_max = prewitt.max()
    if prewitt_max > 0:
        prewitt = np.uint8(prewitt / prewitt_max * 255)
    else:
        prewitt = np.uint8(prewitt)
    return prewitt

def process_image(image, algorithm, **params):
    if algorithm == "Canny":
        return apply_canny(image, params['threshold1'], params['threshold2'])
    elif algorithm == "Sobel":
        return apply_sobel(image, params['ksize'])
    elif algorithm == "Laplacian":
        return apply_laplacian(image, params['ksize'])
    elif algorithm == "Prewitt":
        return apply_prewitt(image)
    return None

def convert_to_downloadable(image):
    is_success, buffer = cv2.imencode(".png", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

st.title("🔍 Edge Detection R&D Project")
st.markdown("### Web-based Computer Vision Edge Detection")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    algorithm = st.selectbox(
        "Edge Detection Algorithm",
        ["Canny", "Sobel", "Laplacian", "Prewitt"]
    )
    
    st.markdown("---")
    st.subheader("Algorithm Parameters")
    
    params = {}
    
    if algorithm == "Canny":
        params['threshold1'] = st.slider("Lower Threshold", 0, 255, 50, 1)
        params['threshold2'] = st.slider("Upper Threshold", 0, 255, 150, 1)
        st.info("**Canny Edge Detection**: Uses a multi-stage algorithm to detect edges with non-maximum suppression.")
        
    elif algorithm == "Sobel":
        params['ksize'] = st.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
        st.info("**Sobel Operator**: Computes gradient magnitude using first-order derivatives.")
        
    elif algorithm == "Laplacian":
        params['ksize'] = st.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3)
        st.info("**Laplacian Operator**: Uses second-order derivatives to detect edges based on zero-crossings.")
        
    elif algorithm == "Prewitt":
        st.info("**Prewitt Operator**: Similar to Sobel but uses a different kernel for gradient computation.")
    
    st.markdown("---")
    st.markdown("""
    **Technologies:**
    - Python + OpenCV
    - Streamlit Web Framework
    - NumPy & PIL
    """)

col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Drag and drop or click to browse"
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    use_sample = st.button("📸 Use Sample Image", use_container_width=True)

original_image = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif use_sample:
    original_image = cv2.imread("attached_assets/stock_images/modern_city_skyline__6ab2b1cf.jpg")

if original_image is not None:
    
    processed_image = process_image(original_image, algorithm, **params)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader(f"🎨 {algorithm} Edge Detection")
        st.image(processed_image, use_container_width=True, channels="GRAY")
    
    st.markdown("---")
    
    col_download1, col_download2, col_info = st.columns([1, 1, 2])
    
    with col_download1:
        img_bytes = convert_to_downloadable(processed_image)
        st.download_button(
            label="📥 Download Edge Detected Image",
            data=img_bytes,
            file_name=f"edge_detected_{algorithm.lower()}.png",
            mime="image/png"
        )
    
    with col_info:
        st.metric("Image Size", f"{original_image.shape[1]} x {original_image.shape[0]} px")
        st.metric("Algorithm", algorithm)

else:
    st.info("👆 Please upload an image to get started with edge detection analysis.")
    
    st.markdown("---")
    st.markdown("""
    ### About Edge Detection Algorithms
    
    **Canny Edge Detector**
    - Multi-stage algorithm with noise reduction
    - Non-maximum suppression
    - Hysteresis thresholding for robust edge detection
    
    **Sobel Operator**
    - First-order derivative approximation
    - Separate horizontal and vertical gradient computation
    - Good for detecting gradual intensity changes
    
    **Laplacian Operator**
    - Second-order derivative detection
    - Isotropic (rotation invariant)
    - Sensitive to noise but detects fine details
    
    **Prewitt Operator**
    - Similar to Sobel with simpler kernels
    - Computes gradient magnitude
    - Less sensitive to noise than Laplacian
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b949e;'>Edge Detection R&D Project | Powered by OpenCV & Streamlit</div>",
    unsafe_allow_html=True
)
