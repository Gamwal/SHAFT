# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:59:10 2022

@author: gamaliel.adun
"""
#%%
import pandas as pd
import pyodbc
import streamlit as st

#%%
data = 'C:/Users/adun.gamaliel/Downloads/data.mdb'
@st.cache
def loader_excel(data):
    return pd.read_excel(data)

@st.cache
def read_access_tables(data):
    #table = st.text_input('Table Name',"This is Case Sensitive")
    path = (r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+data+'; ')
    conn = pyodbc.connect(path)
    #sql = f"select * from {table}"
    sql2 = "select * from MONTHLYPROD where UNIQUEID like 'BONT%'"
    return pd.read_sql(sql2,conn)
