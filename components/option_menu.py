import streamlit as st
from streamlit_option_menu import option_menu

def create_option_menu():
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",  
            options=["Beranda", "Proses Klasifikasi", "Tentang Kami"],  
            icons=["house", "file-earmark-text", "graph-up-arrow"], 
            menu_icon="cast",  
            default_index=0,  
            orientation="vertical" 
        )
    return selected