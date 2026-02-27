import streamlit as st
import numpy as np
from scipy.stats import t
from statistics import stdev
 
st.title("Two Sample t-Test Calculator")
 
# User Inputs
st.write("Enter Sample 1 values (comma separated)")
a_input = st.text_input("Sample 1")
 
st.write("Enter Sample 2 values (comma separated)")
b_input = st.text_input("Sample 2")
 
alpha = st.number_input("Significance Level (alpha)", value=0.05)
 
alt = st.selectbox(
    "Alternative Hypothesis",
    ("two-sided", "greater", "lesser")
)
 
if st.button("Calculate"):
 
    try:
        a = list(map(float, a_input.split(",")))
        b = list(map(float, b_input.split(",")))
 
        n1 = len(a)
        n2 = len(b)
 
        x1 = np.mean(a)
        x2 = np.mean(b)
 
        sd1 = stdev(a)
        sd2 = stdev(b)
 
        se = np.sqrt((sd1**2 / n1) + (sd2**2 / n2))
        t_cal = (x1 - x2) / se
 
        df = n1 + n2 - 2
 
        if alt == "two-sided":
            t_pos = t.ppf(1 - alpha/2, df)
            t_neg = t.ppf(alpha/2, df)
            p = 2 * (1 - t.cdf(abs(t_cal), df))
 
            st.write("t calculated:", t_cal)
            st.write("t positive critical:", t_pos)
            st.write("t negative critical:", t_neg)
            st.write("p-value:", p)
 
        elif alt == "greater":
            t_pos = t.ppf(1 - alpha, df)
            p = 1 - t.cdf(t_cal, df)
 
            st.write("t calculated:", t_cal)
            st.write("t positive critical:", t_pos)
            st.write("p-value:", p)
 
        elif alt == "lesser":
            t_neg = t.ppf(alpha, df)
            p = t.cdf(t_cal, df)
 
            st.write("t calculated:", t_cal)
            st.write("t negative critical:", t_neg)
            st.write("p-value:", p)
 
    except:
        st.error("Please enter valid numeric values separated by commas.")