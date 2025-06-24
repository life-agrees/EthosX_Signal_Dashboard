# app.py
import streamlit as st
from archive.dashboard import show_dashboard


def main():
    st.set_page_config(page_title="EthosX Predictive Dashboard", layout="wide")
    show_dashboard()


if __name__ == "__main__":
    main()