import streamlit as st
from PIL import Image


def main():
    st.title("Compiladores 2, Proyecto 2")
    st.subheader("Data Science")

    image = Image.open('images/inicio.jpg')
    st.image(image, caption='imagen random')


if __name__ == "__main__":
    main()