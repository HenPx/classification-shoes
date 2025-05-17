import streamlit as st
import pickle
from components.option_menu import create_option_menu
from pages_control.home import create_home_menu
from pages_control.about import create_about_menu
from pages_control.classification import create_classification_menu
from utils.svm_rbf_manual import ManualSVM


# load the model
ManualSVM()
with open('utils/svmmodel (2).pkl', 'rb') as file:
    model = pickle.load(file)
with open("utils/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("utils/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
    

def main():
    
    # Initialize session state
    if 'open_selected_image' not in st.session_state:
        st.session_state['open_selected_image'] = None
    if 'open_selected_image_zip' not in st.session_state:
        st.session_state['open_selected_image_zip'] = None
    if 'preprocessing' not in st.session_state:
        st.session_state['preprocessing'] = None
    if 'classification' not in st.session_state:
        st.session_state['classification'] = None
    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files'] = []
    if 'get_label' not in st.session_state:
        st.session_state['get_label'] = []
    if "extracted_files" not in st.session_state:
        st.session_state['extracted_files'] = []
    if 'start_machine' not in st.session_state:
        st.session_state['start_machine'] = True

    
    st.markdown("<h1 style='text-align: center;'>Klasifikasi Gambar SVM dan GLCM</h1>", unsafe_allow_html=True)
    # center title with markdown
    st.markdown("<p style='text-align: center;'>Aplikasi ini dibuat sepenuhnya oleh: Kelompok 5 Kelas B</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.sidebar:
        selected = create_option_menu()

    if selected == "Beranda":
        create_home_menu()
    elif selected == "Proses Klasifikasi":
        create_classification_menu(model, scaler, le)
    elif selected == "Tentang Kami":  
        create_about_menu()
            
if __name__ == "__main__":
    main()
    
