#Import All the Required Libraries
import streamlit as st
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import torch

#Get the absolute path of the current file
FILE = Path(__file__).resolve()

#Get the parent directory of the current file
ROOT = FILE.parent

#Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

#Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

#Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'UserImage.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'DetectedImage.jpg'

#Model Configurations
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'my_model.pt'

#Page Layout
st.set_page_config(
    page_title = "Ultra Detect",
    page_icon = "🏥"
)

#Header
st.header("NT and NB Detection using YOLO11")

#SideBar
st.sidebar.header("Model Configurations")

model_type = "Detection"

#Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40))/100

model_path = Path(DETECTION_MODEL)

#Load the YOLO Model
try:
    model = YOLO(model_path)
    class_names = model.names
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(e)

#Image Configuration
st.sidebar.header("Image Configuration")

source_image = st.sidebar.file_uploader(
    "Choose an Image....", type = ("jpg", "png", "jpeg", "bmp", "webp")
)

col1, col2 = st.columns(2)
with col1:
    try:
        if source_image is None:
            default_image_path = str(DEFAULT_IMAGE)
            default_image = Image.open(default_image_path)
            st.image(default_image_path, caption = "Default Image", use_container_width=True)
        else:
            uploaded_image  =Image.open(source_image)
            st.image(source_image, caption = "Uploaded Image", use_container_width = True)
    except Exception as e:
        st.error("Error Occured While Opening the Image")
        st.error(e)
with col2:
    try:
        if source_image is None:
            default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
            default_detected_image = Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption = "Detected Image", use_container_width = True)
        else:
            if st.sidebar.button("Detect Objects"):
                result = model.predict(uploaded_image, conf = confidence_value)
                boxes = result[0].boxes
                result_plotted = result[0].plot()[:,:,::-1]
                st.image(result_plotted, caption = "Detected Image", use_container_width = True)

                try:
                    with st.expander("Detection Results"):
                        image_width, image_height = uploaded_image.size
                        for box in boxes:
                            x, y, w, h = box.xywh[0]
                            class_id = int(box.cls.item())
                            class_name = class_names[class_id] if class_id in class_names else "Unknown"

                            # Hitung koordinat relatif
                            x_fix = x / image_width
                            y_fix = y / image_height
                            w_fix = w / image_width
                            h_fix = h / image_height

                            relative_box = torch.tensor([x_fix, y_fix, w_fix, h_fix,
                                                         box.conf.item(), box.cls.item()],
                                                        dtype=torch.float32)

                            st.markdown(f"**Class:** {class_name} <br>{relative_box}", unsafe_allow_html=True)
                except Exception as e:
                    st.error(e)
    except Exception as e:
        st.error("Error Occured While Opening the Image")
        st.error(e)
