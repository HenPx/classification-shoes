import streamlit as st
import pickle
from components.option_menu import create_option_menu
from pages_control.home import create_home_menu
from pages_control.about import create_about_menu
from pages_control.classification import create_classification_menu
# load the model
with open('utils/svm_rbf.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Extract model parameters
alpha = model["alpha"]
b = model["b"]
X_train = model["X_train"]
y_train = model["y_train"]
gamma = model["gamma"]

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
    st.markdown("<p style='text-align: center;'>Aplikasi ini dibuat sepenuhnya oleh: NAMA_PEMBUAT</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.sidebar:
        selected = create_option_menu()

    if selected == "Beranda":
        create_home_menu()
    elif selected == "Proses Klasifikasi":
        create_classification_menu(model, alpha, b, X_train, y_train, gamma)
    elif selected == "Tentang Kami":  
        create_about_menu()
            
if __name__ == "__main__":
    main()
    
    

# with open('utils/svm_ini_brow.pkl', 'rb') as file:
#     model_baru = pickle.load(file)


# value_predict = st.text_input("Masukkan nilai prediksi", "0")

# # buat value menjadi inputan array 331.3830586	6.060346949	0.578375676	0.200768492	0.448071972	0.981726384	947.6120652	11.14669229	0.543731208	0.193243306	0.439594479	0.947920049	380.4293799	7.262549213	0.579570426	0.201623395	0.449024938	0.979021098	303.006448	6.285510571	0.557020349	0.19485555	0.441424455	0.983347024
# split = []
# split = value_predict.split("\t")
# # ubah string jadi array numpy dan reshape
# st.write(split)
# print(split)

# test = np.array(split, dtype=float)
# test = test.reshape(1, -1)

# categories = ["Boot", "Sandal", "Shoe"]  # Jangan pakai 'class'

# # Prediksi saat tombol diklik
# if st.button("Prediksi", key="but_predict"):
#     prediction = model_baru.predict(test)  # pastikan model_baru sudah di-load
#     predicted_class = categories[prediction[0]]  # indeks ke nama kelas
#     st.write(f"Prediksi: {predicted_class}")
    
