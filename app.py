import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Urban Feature Mapping",
    page_icon="🚁",
    layout="wide"
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("⚙️ Analysis Settings")

    st.write("**Processing Pipeline**")

    st.checkbox(
        "Image Preprocessing",
        value=True,
        disabled=True
    )

    st.checkbox(
        "Vegetation Detection",
        value=True,
        disabled=True
    )

    st.checkbox(
        "Road Detection",
        value=True,
        disabled=True
    )

    st.checkbox(
        "Structure Detection",
        value=True,
        disabled=True
    )

    st.divider()

    st.caption(
        "This application is an experimental "
        "computer-vision based MVP."
    )

# ==========================================
# HEADER
# ==========================================

st.title("🚁 Urban Feature Mapping")

st.markdown(
    "### AI-Based Drone Imagery Analysis"
)

st.write(
    "Analyze aerial imagery and generate a visual map "
    "of vegetation, road structures and built-up regions."
)

st.warning(
    "⚠️ Experimental MVP — This system is not a legally "
    "accurate cadastral or land-survey system."
)

# ==========================================
# UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload Drone / Aerial Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ======================================
    # LOAD IMAGE
    # ======================================

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_np = np.array(image)

    # ======================================
    # PREPROCESSING
    # ======================================

    resized = cv2.resize(
        image_np,
        (800, 600)
    )

    denoised = cv2.GaussianBlur(
        resized,
        (5, 5),
        0
    )

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    processed = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )

    # ======================================
    # VEGETATION
    # ======================================

    hsv = cv2.cvtColor(
        processed,
        cv2.COLOR_RGB2HSV
    )

    lower_green = np.array(
        [25, 40, 30]
    )

    upper_green = np.array(
        [95, 255, 255]
    )

    vegetation_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    vegetation_result = cv2.bitwise_and(
        processed,
        processed,
        mask=vegetation_mask
    )

    vegetation_percentage = (
        np.count_nonzero(
            vegetation_mask
        ) / vegetation_mask.size
    ) * 100

    # ======================================
    # ROAD DETECTION
    # ======================================

    gray = cv2.cvtColor(
        processed,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(
        gray,
        80,
        160
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    road_mask = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    road_result = cv2.bitwise_and(
        processed,
        processed,
        mask=road_mask
    )

    # ======================================
    # STRUCTURE DETECTION
    # ======================================

    building_gray = cv2.cvtColor(
        processed,
        cv2.COLOR_RGB2GRAY
    )

    _, building_mask = cv2.threshold(
        building_gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    building_kernel = np.ones(
        (7, 7),
        np.uint8
    )

    building_mask = cv2.morphologyEx(
        building_mask,
        cv2.MORPH_CLOSE,
        building_kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        building_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_building_mask = np.zeros_like(
        building_mask
    )

    building_count = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area > 500:

            cv2.drawContours(
                clean_building_mask,
                [contour],
                -1,
                255,
                -1
            )

            building_count += 1

    building_result = cv2.bitwise_and(
        processed,
        processed,
        mask=clean_building_mask
    )

    # ======================================
    # FINAL MAP
    # ======================================

    final_map = processed.copy()

    vegetation_overlay = np.zeros_like(
        final_map
    )

    vegetation_overlay[
        vegetation_mask > 0
    ] = [0, 180, 0]

    final_map = cv2.addWeighted(
        final_map,
        0.65,
        vegetation_overlay,
        0.35,
        0
    )

    road_overlay = np.zeros_like(
        final_map
    )

    road_overlay[
        road_mask > 0
    ] = [255, 140, 0]

    final_map = cv2.addWeighted(
        final_map,
        0.75,
        road_overlay,
        0.25,
        0
    )

    building_overlay = np.zeros_like(
        final_map
    )

    building_overlay[
        clean_building_mask > 0
    ] = [220, 0, 0]

    final_map = cv2.addWeighted(
        final_map,
        0.75,
        building_overlay,
        0.25,
        0
    )

    # ======================================
    # RESULTS
    # ======================================

    st.success(
        "✅ Analysis completed successfully!"
    )

    # --------------------------------------
    # IMAGE COMPARISON
    # --------------------------------------

    st.header("🖼️ Image Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Original Drone Image",
            use_container_width=True
        )

    with col2:
        st.image(
            processed,
            caption="Preprocessed Image",
            use_container_width=True
        )

    # --------------------------------------
    # FEATURE DETECTION
    # --------------------------------------

    st.header("🔍 Feature Detection")

    tab1, tab2, tab3 = st.tabs(
        [
            "🌳 Vegetation",
            "🛣️ Roads",
            "🏢 Structures"
        ]
    )

    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                vegetation_mask,
                caption="Vegetation Mask",
                use_container_width=True
            )

        with col2:
            st.image(
                vegetation_result,
                caption="Detected Vegetation",
                use_container_width=True
            )

        st.metric(
            "Estimated Vegetation Coverage",
            f"{vegetation_percentage:.2f}%"
        )

    with tab2:

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                road_mask,
                caption="Road / Edge Mask",
                use_container_width=True
            )

        with col2:
            st.image(
                road_result,
                caption="Detected Road Features",
                use_container_width=True
            )

    with tab3:

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                clean_building_mask,
                caption="Structure Mask",
                use_container_width=True
            )

        with col2:
            st.image(
                building_result,
                caption="Detected Structures",
                use_container_width=True
            )

        st.metric(
            "Detected Large Structures",
            building_count
        )

    # --------------------------------------
    # FINAL MAP
    # --------------------------------------

    st.header("🗺️ Final Urban Feature Map")

    st.image(
        final_map,
        caption="Combined Feature Visualization",
        use_container_width=True
    )

    st.markdown(
        """
        **Legend**

        🟢 Vegetation &nbsp;&nbsp; 
        🟠 Road / Linear Features &nbsp;&nbsp; 
        🔴 Structures
        """
    )

    # --------------------------------------
    # SUMMARY
    # --------------------------------------

    st.header("📊 Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Vegetation",
            f"{vegetation_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Structures",
            building_count
        )

    with col3:
        st.metric(
            "Processed Resolution",
            "800 × 600"
        )

    # --------------------------------------
    # DOWNLOAD
    # --------------------------------------

    final_map_rgb = cv2.cvtColor(
        final_map,
        cv2.COLOR_BGR2RGB
    )

    success, encoded_image = cv2.imencode(
        ".png",
        final_map_rgb
    )

    if success:

        st.download_button(
            "⬇️ Download Urban Feature Map",
            data=encoded_image.tobytes(),
            file_name="urban_feature_map.png",
            mime="image/png"
        )

else:

    st.info(
        "👆 Upload an aerial image above to start analysis."
    )