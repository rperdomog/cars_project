import streamlit as st
import pandas as pd 
import plotly_express as px

df = pd.read_csv("vehicles_us.csv")
df['manufacturer'] = df['model'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)

st.header("Cars project sprint 4")
st.dataframe(df)

#Plot vehicles types per manufacturer
st.header("Vehicles types by manufacturer")
fig = px.histogram(df, x='manufacturer', color='type')
st.write(fig)

st.header('Conditions vs model year')
fig2 = px.histogram(df, x='model_year', color='condition')
st.write(fig2)