import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Object Detection App")
st.markdown("Detect objects in any image using "
            "YOLOv8 — the world's most powerful "
            "real-time object detector.")
st.markdown("---")

# Load YOLO model
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return model

with st.spinner("Loading YOLOv8 model..."):
    model = load_model()
st.success("✅ YOLOv8 model loaded!")

# COCO class colors
COLORS = [
    '#e74c3c', '#3498db', '#2ecc71',
    '#f39c12', '#9b59b6', '#1abc9c',
    '#e67e22', '#27ae60', '#2980b9',
    '#8e44ad', '#16a085', '#d35400',
    '#c0392b', '#2c3e50', '#7f8c8d',
    '#f1c40f', '#e91e63', '#00bcd4',
    '#ff5722', '#795548'
]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(
        int(hex_color[i:i+2], 16)
        for i in (0, 2, 4)
    )

def detect_objects(image, conf_threshold):
    results = model(
        image,
        conf=conf_threshold,
        verbose=False
    )
    return results

def draw_detections(image, results,
                    show_labels,
                    show_confidence):
    draw      = ImageDraw.Draw(image)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = \
                box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]

            color = COLORS[cls % len(COLORS)]
            rgb   = hex_to_rgb(color)

            # Draw box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=rgb,
                width=3
            )

            # Draw label
            if show_labels:
                text = label
                if show_confidence:
                    text += f" {conf:.0%}"

                text_bbox = draw.textbbox(
                    (x1, y1 - 20),
                    text
                )
                draw.rectangle(
                    [text_bbox[0] - 2,
                     text_bbox[1] - 2,
                     text_bbox[2] + 2,
                     text_bbox[3] + 2],
                    fill=rgb
                )
                draw.text(
                    (x1, y1 - 20),
                    text,
                    fill=(255, 255, 255)
                )

            detections.append({
                'Object':     label,
                'Confidence': f"{conf:.1%}",
                'X1': int(x1), 'Y1': int(y1),
                'X2': int(x2), 'Y2': int(y2),
                'Width':  int(x2 - x1),
                'Height': int(y2 - y1)
            })

    return image, detections

# Sidebar
st.sidebar.header("⚙️ Detection Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    0.1, 0.9, 0.25, 0.05,
    help="Minimum confidence to show detection"
)
show_labels     = st.sidebar.checkbox(
    "Show labels", value=True)
show_confidence = st.sidebar.checkbox(
    "Show confidence %", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("""
**YOLOv8 Nano (yolov8n)**
- 80 object categories
- COCO dataset trained
- Real-time detection speed
- 3.2M parameters
- Best for CPU inference
""")

st.sidebar.markdown("### 🎯 What it detects")
categories = [
    "👤 Person", "🚗 Car", "🚌 Bus",
    "🚲 Bicycle", "🐕 Dog", "🐈 Cat",
    "📱 Phone", "💻 Laptop", "🪑 Chair",
    "🍕 Food items", "✈️ Airplane",
    "🚢 Boat", "🌿 Plants", "80 total..."
]
for cat in categories:
    st.sidebar.markdown(f"• {cat}")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📷 Upload & Detect",
    "📊 Detection Stats",
    "📚 About YOLO"
])

# Session state for detections
if 'last_detections' not in st.session_state:
    st.session_state.last_detections = []
if 'last_image' not in st.session_state:
    st.session_state.last_image = None

# Tab 1 — Upload
with tab1:
    st.markdown("### Upload Image for Detection")

    uploaded = st.file_uploader(
        "Upload an image:",
        type=['jpg', 'jpeg', 'png',
              'webp', 'bmp']
    )

    if uploaded:
        image = Image.open(uploaded)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📷 Original")
            st.image(image,
                     use_column_width=True)
            st.caption(
                f"Size: {image.size[0]}×"
                f"{image.size[1]} | "
                f"Mode: {image.mode}")

        with col2:
            st.markdown("#### 🎯 Detected")
            with st.spinner(
                "Running YOLOv8..."):
                results = detect_objects(
                    image, conf_threshold)
                result_image = image.copy()
                result_image, detections = \
                    draw_detections(
                        result_image,
                        results,
                        show_labels,
                        show_confidence
                    )

            st.image(result_image,
                     use_column_width=True)

            if detections:
                st.success(
                    f"✅ {len(detections)} "
                    f"object(s) detected!")
            else:
                st.warning(
                    "No objects detected. "
                    "Try lowering the "
                    "confidence threshold.")

            st.session_state\
                .last_detections = detections
            st.session_state\
                .last_image = result_image

        if detections:
            st.markdown("---")
            st.markdown(
                "### 📋 Detection Results")

            import pandas as pd
            det_df = pd.DataFrame(detections)
            st.dataframe(
                det_df,
                use_container_width=True,
                hide_index=True)

            # Object count
            from collections import Counter
            obj_counts = Counter(
                d['Object']
                for d in detections)

            st.markdown(
                "### 📊 Objects Found")
            cols = st.columns(
                min(len(obj_counts), 4))
            for i, (obj, count) in \
                    enumerate(
                        obj_counts.items()):
                with cols[i % 4]:
                    st.metric(
                        obj.capitalize(),
                        count)

# Tab 2 — Stats
with tab2:
    st.markdown("### 📊 Detection Statistics")

    if st.session_state.last_detections:
        import pandas as pd
        import matplotlib.pyplot as plt
        from collections import Counter

        detections = \
            st.session_state.last_detections

        obj_counts = Counter(
            d['Object']
            for d in detections)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "#### Object Distribution")
            fig, ax = plt.subplots(
                figsize=(7, 4))
            colors = [
                COLORS[i % len(COLORS)]
                for i in range(
                    len(obj_counts))
            ]
            ax.bar(
                obj_counts.keys(),
                obj_counts.values(),
                color=colors,
                edgecolor='black'
            )
            ax.set_title(
                'Detected Objects Count',
                fontsize=12)
            ax.set_ylabel('Count')
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown(
                "#### Confidence Scores")
            confs = [
                float(d['Confidence']
                      .replace('%', '')) / 100
                for d in detections
            ]
            labels = [
                d['Object']
                for d in detections
            ]

            fig2, ax2 = plt.subplots(
                figsize=(7, 4))
            bar_colors = [
                '#2ecc71' if c >= 0.7
                else '#f39c12' if c >= 0.5
                else '#e74c3c'
                for c in confs
            ]
            ax2.barh(
                range(len(labels)),
                confs,
                color=bar_colors,
                edgecolor='black'
            )
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels)
            ax2.set_xlim(0, 1)
            ax2.axvline(
                x=conf_threshold,
                color='red',
                linestyle='--',
                label=f'Threshold '
                      f'({conf_threshold})'
            )
            ax2.set_title(
                'Confidence per Detection',
                fontsize=12)
            ax2.set_xlabel('Confidence')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Objects",
                  len(detections))
        s2.metric("Unique Classes",
                  len(obj_counts))
        s3.metric("Avg Confidence",
                  f"{np.mean(confs):.1%}")
        s4.metric("Max Confidence",
                  f"{max(confs):.1%}")
    else:
        st.info(
            "Upload and detect an image "
            "first to see statistics!")

# Tab 3 — About
with tab3:
    st.markdown("### 📚 About YOLO")
    st.markdown("""
    #### What is YOLO?
    **YOLO (You Only Look Once)** is a
    state-of-the-art real-time object
    detection algorithm created by
    Joseph Redmon in 2015.

    Unlike traditional methods that scan
    an image multiple times, YOLO looks
    at the entire image **once** and
    predicts all bounding boxes and
    class probabilities simultaneously.

    #### Why YOLOv8?
    - **Fastest** model for real-time detection
    - **Most accurate** in its size class
    - **Easy to use** via Ultralytics API
    - Trained on **COCO dataset** — 80 classes
    - Used in **production** worldwide

    #### Applications
    - 🚗 Self-driving cars
    - 🏥 Medical image analysis
    - 🏭 Manufacturing defect detection
    - 🔒 Security and surveillance
    - 🛒 Retail analytics
    - 🌾 Agricultural monitoring
    """)

    import pandas as pd
    models_df = pd.DataFrame({
        'Model':      ['YOLOv8n', 'YOLOv8s',
                       'YOLOv8m', 'YOLOv8l',
                       'YOLOv8x'],
        'Size':       ['3.2M', '11.2M',
                       '25.9M', '43.7M',
                       '68.2M'],
        'mAP50':      ['37.3', '44.9',
                       '50.2', '52.9',
                       '53.9'],
        'Speed (ms)': ['0.99', '1.20',
                       '1.83', '2.39',
                       '3.53'],
        'Best for':   ['CPU/Edge', 'Balanced',
                       'Accuracy', 'High Acc',
                       'Max Acc']
    })
    st.markdown("#### YOLOv8 Model Variants")
    st.dataframe(models_df,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Object Detection using YOLOv8 | "
    "Powered by Ultralytics"
)