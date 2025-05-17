import streamlit as st
# from PIL import Image
from plotly import graph_objs as go

def create_home_menu():
    st.markdown("""
    ### 👋 Selamat Datang di Aplikasi Klasifikasi!

    Aplikasi ini dirancang untuk mengklasifikasikan tiga jenis dataset gambar:
    - **Shoes** 🥿
    - **Boots** 👢
    - **Sandal** 🩴

    Algoritma yang digunakan adalah **Support Vector Machine (SVM)** dengan teknik **Ekstraksi Fitur GLCM** (Gray Level Co-occurrence Matrix).

    📊 **Yuk, kita kenalan dulu dengan datasetnya:**
    """)

    # Visualisasi Dataset
    labels = ['Shoes', 'Boots', 'Sandal']
    values = [5000, 5000, 5000]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=['#dc2e17', '#0a3633', '#126b76']))])

    fig.update_layout(title_text="Distribusi Dataset (Balance)", title_x=0.5)
    st.plotly_chart(fig)

    st.markdown("""
    🧐 **Dataset yang digunakan seimbang**, masing-masing kategori memiliki jumlah data yang sama, yaitu 5000 gambar. 
    Berikut adalah contoh data yang akan digunakan untuk pelatihan model:
    """)

    # Contoh Gambar
    col1, col2, col3 = st.columns(3)

    with col1:
        
        # image_sandal = Image.open("images/sandal.jpg")
        st.image("images/sandal.jpg", caption="Sandal")

    with col2:
        # image_shoes = Image.open("images/shoe.jpg")
        st.image("images/shoe.jpg", caption="Shoes")

    with col3:
        # image_boots = Image.open("images/boot.jpg")
        st.image("images/boot.jpg", caption="Boots")

    st.markdown("""
    ✨ **Selamat menjelajahi aplikasi ini!** Jangan ragu untuk mencoba berbagai fitur dan lihat hasil klasifikasi dataset sepatu, boots, dan sandal.
    """)