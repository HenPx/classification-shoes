import streamlit as st
import numpy as np
from PIL import Image
import cv2
from zipfile import ZipFile
from io import BytesIO
from utils.feat_glcm import compute_glcm, extract_glcm_features
from utils.rbf_svm import predict_svm_rbf


def extract_zip_to_state(zip_file):
    uploaded_files = []
    with ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                with z.open(file_name) as file:
                    # Simpan sebagai BytesIO agar dapat diakses berulang kali
                    file_data = BytesIO(file.read())
                    uploaded_files.append(file_data)
    st.session_state['uploaded_files'] = uploaded_files

def create_classification_menu(model, alpha, b, X_train, y_train, gamma):
    #load model
    model = model
    
    st.subheader("🧠 Klasifikasi Menggunakan Algoritma SVM dan GLCM")

    st.markdown("""
    🎯 **Ayo coba lakukan klasifikasi!**

    Silakan masukkan data klasifikasi Anda untuk diproses dengan model yang telah kami buat. Tahapannya adalah sebagai berikut:

    1️⃣ **Upload data** 📂
    
    2️⃣ **Ekstraksi fitur akan muncul secara otomatis** 🛠️
    
    3️⃣ **Prediksi data akan ditampilkan** 🔍
    
    **Note:** Jika tombol upload tidak ada atau ingin melakukan perubahan silahkan tekan tombol Reset.
    """)
    tabs = st.tabs(["Upload Gambar", "Upload Zip"])
    
    with tabs[0]:
        if st.session_state['start_machine']:
            if st.button("Upload Gambar", key="upload_gambar"):
                st.session_state['open_selected_image'] = True
                st.session_state['start_machine'] = False
        
        if st.session_state['open_selected_image']:
            uploaded_files = st.file_uploader(
                "Choose files", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.session_state['uploaded_files'] = uploaded_files
                st.write(f"{len(uploaded_files)} files uploaded successfully!")
                
                if st.button("Mulai Preprocessing", key="but_start_preprocessing"):
                    st.session_state['open_selected_image'] = False
                    st.session_state['preprocessing'] = True
                    # Extract GLCM features for each image
                    cols_per_row = 7  # Jumlah gambar per baris
                    files = st.session_state['uploaded_files']
                    
                    for i in range(0, len(files), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, file in enumerate(files[i:i+cols_per_row]):
                            with cols[j]:
                                image = Image.open(file)
                                image = np.array(image)
                                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                gray = cv2.resize(gray, (128, 128))
                                gray = cv2.equalizeHist(gray)
                                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                                st.image(gray)
                                gray = gray.astype(np.float32) / 255.0
                                gray = (gray * 255).astype(np.uint8)

                                glcm = compute_glcm(gray)
                                features = extract_glcm_features(glcm)
                                st.write("Fitur GLCM:")
                                st.write(", ".join([f"{v:.3f}" for v in features.flatten()]))
                                features = features.reshape(1, -1)

                                prediction = predict_svm_rbf(X_train, y_train, features, alpha, b, gamma)
                                label = "Boot" if prediction[0] == 1 else "Not Boot"
                                st.session_state['get_label'].append(label)
                    
    with tabs [1]:
        if st.session_state['start_machine']:
            if st.button("Upload Zip", key="upload_gambar_1"):
                st.session_state['open_selected_image_zip'] = True
                st.session_state['start_machine'] = False
        
        if st.session_state['open_selected_image_zip']:
            st.session_state['start_machine'] = False
            zip_file = st.file_uploader(
                "Choose files", 
                type="zip", 
                accept_multiple_files=False
            )
            
            if zip_file:
                extract_zip_to_state(zip_file)
                st.write(f"{len(st.session_state['uploaded_files'])} files extracted and stored in state!")
                
                
                if st.button("Mulai Preprocessing", key="but_start_preprocessing_1"):
                    st.write("Preprocessing started!")
                    st.session_state['open_selected_image_zip'] = False
                    st.session_state['preprocessing'] = True
                    cols_per_row = 7  
                    files = st.session_state['uploaded_files']
                    
                    for i in range(0, len(files), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, file in enumerate(files[i:i+cols_per_row]):
                            with cols[j]:
                                image = Image.open(file)
                                image = np.array(image)
                                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                                gray = cv2.resize(gray, (128, 128))
                                gray = cv2.equalizeHist(gray)
                                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                                st.image(gray)
                                gray = gray.astype(np.float32) / 255.0
                                gray = (gray * 255).astype(np.uint8)

                                glcm = compute_glcm(gray)
                                features = extract_glcm_features(glcm)
                                st.write("Fitur GLCM:")
                                st.write(", ".join([f"{v:.3f}" for v in features.flatten()]))
                                features = features.reshape(1, -1)
                                prediction = predict_svm_rbf(X_train, y_train, features, alpha, b, gamma)
                                label = "Boot" if prediction[0] == 1 else "Not Boot"
                                st.session_state['get_label'].append(label)
    
    if st.session_state['preprocessing']:
        st.session_state['open_selected_image'] = False
        st.session_state['open_selected_image_zip'] = False
        if st.button("Mulai Klasifikasi", key="but_start_classification"):
            st.write("Classification started!")
            st.session_state['preprocessing'] = False
            st.session_state['classification'] = True
            
            
    if st.session_state['classification']:
        
        if 'uploaded_files' in st.session_state and 'get_label' in st.session_state:
            num_files = len(st.session_state['uploaded_files'])
            num_labels = len(st.session_state['get_label'])

            if num_files != num_labels:
                st.warning("Jumlah label tidak sesuai dengan jumlah gambar! Silahkan tekan tombol reset.")
            else:
                st.write("Klasifikasi telah dilakukan! Silahkan tekan tombol Reset untuk mengulangi proses.")

                cols_per_row = 7  
                for i in range(0, num_files, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, (file, label) in enumerate(zip(
                        st.session_state['uploaded_files'][i:i+cols_per_row],
                        st.session_state['get_label'][i:i+cols_per_row]
                    )):
                        with cols[j]:
                            st.image(file, caption=f"Prediksi: {label}", use_container_width=True)
        else:
            st.warning("Tidak ada gambar yang akan diproses.")

            
    if st.button("Reset", key="but_reset"):
        st.session_state['classification'] = False
        st.session_state['uploaded_files'] = []
        st.session_state['get_label'] = []
        st.session_state['open_selected_image'] = False
        st.session_state['open_selected_image_zip'] = False
        st.session_state['preprocessing'] = False
        st.session_state['classification'] = False
        st.session_state['start_machine'] = True
        st.success("Reset successful!")
        st.rerun()