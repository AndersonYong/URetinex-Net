import streamlit as st

from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Low Light Image Enhancement")

st.write("## Enhance your low light image")
st.write(
    ":dog: Try uploading an image to watch the enhanced image. Full quality images can be downloaded from the sidebar.:grin:"
)
st.sidebar.write("## Upload and download :gear:")


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    # fixed = image_adaptive_lut_evaluation.runforstreamlit(image)
    fixed = evaluation_LOL.runforstreamlit(image)
    col2.write("Ehanced Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download ehanced image", convert_image(fixed), "ehanced.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("./demo_images/3.png")
