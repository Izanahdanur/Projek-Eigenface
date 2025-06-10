import streamlit as st
import time
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
import zipfile

from eigenface_utils import (
    load_images_from_folder,
    compute_mean_face,
    compute_top_k_eigenfaces,
    recognize
)

IMG_SIZE = (100, 100)
K = 10  # Jumlah eigenface

st.set_page_config(page_title="Face Recognition App", layout="wide")

def main():
    st.markdown("<h1 style='text-align: center;'> Face Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar threshold slider
    st.sidebar.title("Pengaturan")
    threshold = st.sidebar.slider("Threshold Euclidean", min_value=1000, max_value=300000, value=100000, step=1000)

    col1, col2, col3 = st.columns([1.5, 2, 2])

    with col1:
        st.markdown("### Insert Your Dataset")
        dataset_folder = st.file_uploader("Choose Folder Zip", type="zip")

        if dataset_folder:
            extract_path = tempfile.mkdtemp()
            with zipfile.ZipFile(dataset_folder, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            dataset_dir = extract_path
        else:
            dataset_dir = None

        st.markdown("### Insert Your Image")
        test_img_file = st.file_uploader("Choose Test Image", type=["jpg", "jpeg", "png"])

        st.markdown("### Result")
        result_text = st.empty()

    with col2:
        st.markdown("### Test Image")
        test_image_display = st.empty()

    with col3:
        st.markdown("### Closest Match")
        result_image_display = st.empty()

    if test_img_file and dataset_dir:
        start_time = time.time()

        temp_img = tempfile.NamedTemporaryFile(delete=False)
        temp_img.write(test_img_file.read())
        test_path = temp_img.name

        # Load dataset
        X, filenames = load_images_from_folder(dataset_dir)
        if X is None or X.shape[1] < K:
            result_text.error("Dataset tidak mencukupi atau tidak valid.")
            return

        mean_face = compute_mean_face(X)
        A = X - mean_face
        eigenfaces = compute_top_k_eigenfaces(A, K)
        projections = eigenfaces.T @ A

        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        if test_img is None:
            result_text.error("Gagal membaca gambar uji. Pastikan file valid dan formatnya .jpg/.png.")
            return

        test_img = cv2.resize(test_img, IMG_SIZE).flatten()

        match_path, dist = recognize(test_img, mean_face, eigenfaces, projections, filenames)

        # Display test image
        test_image_display.image(test_path, caption="Test Image", use_container_width=True)

        # Show result
        if dist < threshold:
            result_image_display.image(match_path, caption="Closest Match", use_container_width=True)
            result_text.markdown(
                f"<p style='color:green;'>Match Found <br>Distance: {dist:.2f}</p>",
                unsafe_allow_html=True
            )
        else:
            result_image_display.image("https://via.placeholder.com/200x200?text=No+Match", caption="No Match", use_container_width=True)
            result_text.markdown(
                f"<p style='color:red;'>No Match br>Distance: {dist:.2f}</p>",
                unsafe_allow_html=True
            )

        # Execution time
        exec_time = time.time() - start_time
        st.markdown(
            f"<p style='text-align: center;'>⏱ Execution time: <span style='color:green;'>{exec_time:.2f} sec</span></p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
