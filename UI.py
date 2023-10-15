# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from main import *
import pandas as pd
import datetime as dt

st.markdown("<h2 style='text-align: center;'>Data Analysis App</h2>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='text-align: left; font-size: 24px;'>Example Files</div>", unsafe_allow_html=True)

amzn_csv_link = "https://raw.githack.com/SoofiiaBoinitska/datasets/main/amzn_stock_data.csv"
st.markdown(f'<a href="{amzn_csv_link}" download style="text-decoration: none;"><button style="margin: 10px; padding: 10px; cursor: pointer; background-color: #F5C573; border: none; color: white; border-radius: 4px;">Download AMZN Stock Data</button></a>', unsafe_allow_html=True)

# Button to download GOOG stock data using HTML button with direct link
goog_csv_link = "https://raw.githack.com/SoofiiaBoinitska/datasets/main/goog_stock_data.csv"
st.markdown(f'<a href="{goog_csv_link}" download style="text-decoration: none;"><button style="margin: 10px; padding: 10px; cursor: pointer; background-color: #F5C573; border: none; color: white; border-radius: 4px;">Download GOOG Stock Data</button></a>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        #df = df[df['symbol'] == selected_ticker]
        #df = df.drop(columns=['symbol'])
        df.set_index('date', inplace=True)
        #df_weekly = df.resample('W').mean()

        st.write("Data Preview:")
        st.table(df.head(10))
        columns = df.columns.tolist()
        selected_column = st.radio("Select a column for analysis:", columns)

        if st.button('Run Analysis'):
            results_df = pd.DataFrame(columns=['model_name', 'R^2', 'SUM(e(k)^2)', 'DW', 'MSE', 'RMSE', 'MAE', 'U', 'model_object'])
            analyzer = TimeSeriesAnalyzer(df[selected_column])
            st.header("Analysis:")
            st.write(analyzer.run_analysis())