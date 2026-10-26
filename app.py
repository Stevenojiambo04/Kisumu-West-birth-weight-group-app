
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Kisumu West Birth Weight Analysis",
    page_icon="👶",
    layout="wide"
)

st.title("The Effect of Gravidity and Parity on Birth Weight of Infants")
st.subheader("Kisumu West Sub-County, Kenya")

st.write("""
This application presents the group's capstone project examining
the relationship between gravidity, parity, and infant birth weight.
""")

st.info(
    "The application will use the group's processed maternity data "
    "for analysis, visualisation, and prediction."
)

st.header("Project Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gravidity")
    st.write("The number of times a woman has been pregnant.")

with col2:
    st.subheader("Parity")
    st.write("Pregnancy outcomes as represented in the study data.")

st.header("Data and Analysis")
st.write(
    "The processed maternity dataset and predictive models "
    "will be connected in the next stage."
)
