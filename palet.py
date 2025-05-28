import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import json
import io

st.set_page_config(page_title="🎨 Color Picker Dominan Ica", layout="centered")

# CSS untuk styling
st.markdown("""
    <style>
    .color-container {
        display: flex;
        justify-content: flex-start;
        flex-wrap: nowrap;
        overflow-x: auto;
        margin-top: 10px;
    }
    .color-box {
        flex: 0 0 auto;
        width: 100px;
        height: 100px;
        margin-right: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        color: #fff;
        font-weight: bold;
        text-shadow: 1px 1px 2px #000;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 Website Color Picker dari Gambar")
st.write(
    "Upload gambar untuk menghasilkan palet warna dengan **5 warna dominan** "
    "dan melihat proporsi warnanya dalam pie chart."
)

uploaded_file = st.file_uploader("Upload gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diupload", use_container_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            st.stop()
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)

    img = np.array(image.resize((200, 200)))
    img_data = img.reshape((-1, 3))

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(img_data)
    colors = np.round(kmeans.cluster_centers_).astype(int)

    # Hitung proporsi setiap cluster
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()

    st.subheader("Palet Warna Dominan")

    hex_colors = []
    color_boxes_html = "<div class='color-container'>"
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        hex_colors.append(hex_color)
        color_boxes_html += f"<div class='color-box' style='background-color:{hex_color}'>{hex_color}</div>"
    color_boxes_html += "</div>"
    st.markdown(color_boxes_html, unsafe_allow_html=True)

    # Tampilkan pie chart proporsi warna
    st.subheader("Proporsi Warna Dominan")
    fig, ax = plt.subplots()
    ax.pie(proportions, labels=hex_colors, colors=hex_colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Buat file JSON untuk download
    palette_data = {
        "palette": [{"rgb": color.tolist(), "hex": hex_colors[idx]} for idx, color in enumerate(colors)],
        "proportions": proportions.tolist()
    }
    json_str = json.dumps(palette_data, indent=4)
    json_bytes = io.BytesIO(json_str.encode())

    st.download_button(
        label="⬇ Download Palet Warna (JSON)",
        data=json_bytes,
        file_name="palette.json",
        mime="application/json"
    )

    st.success("🎉 Warna dominan berhasil dihasilkan!")
else:
    st.info("Silakan upload gambar untuk memulai.")
