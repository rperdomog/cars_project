import streamlit as st
import pandas as pd
import plotly_express as px
import matplotlib as plt
import numpy as np

df = pd.read_csv("vehicles_us.csv")
df = df.dropna()
df['manufacturer'] = df['model'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)

st.header("Final project sprint 4 \n")
st.header("Cars project")
st.dataframe(df)

#Plot vehicles types per manufacturer
st.header("Vehicles types by manufacturer")
fig = px.histogram(df, x='manufacturer', color='type')
st.write(fig)

st.header('Conditions vs model year')
fig2 = px.histogram(df, x='model_year', color='condition')
st.write(fig2)

st.header('Compare price distribution between manufacturers')
manu_list = sorted(df['manufacturer'].unique())
manufacturer_1 = st.selectbox('Select manufacturer 1',
                              manu_list, index=manu_list.index('acura'))

manufacturer_2 = st.selectbox('Select manufacturer 2',
                              manu_list, index=manu_list.index('honda'))
mask_filter = (df['manufacturer'] == manufacturer_1) | (df['manufacturer'] == manufacturer_2)
df_filtered = df[mask_filter]
normalize = st.checkbox('Normalize histogram', value=True)
if normalize:
    histnorm = 'percent'
else:
    histnorm = None
st.write(px.histogram(df_filtered,
                      x='price',
                      nbins=30,
                      color='manufacturer',
                      histnorm=histnorm,
                      barmode='overlay'))

#Brand popularity
df['manufacturer'] = df['model'].str.split(' ').str[0]

top_brands = df['manufacturer'].value_counts().head(20).index
filtered_df = df[df['manufacturer'].isin(top_brands)]


# Car model popularity histogram with dropdown
def plot_model_popularity(df, selected_models=None):
    if selected_models is not None and 'All' not in selected_models:
        filtered_df = df[df['model'].isin(selected_models)]
        title = f"Selected Car Models ({len(selected_models)})"
    else:
        filtered_df = df
        title = 'All Car Models'
    
    model_counts = filtered_df['model'].value_counts()
    fig = px.bar(x=model_counts.index, y=model_counts.values, labels={'x':'Model', 'y':'Count'},
                 title=title, color_discrete_sequence=['#007F73'])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

# Dropdown list with all car models
all_models = df['model'].unique()
default_models = all_models[:5]  # Default to the first two models
selected_models = st.multiselect('Select car models', ['All'] + all_models.tolist(), default=default_models)
plot_model_popularity(df, selected_models)

#Mean brand price

mean_price_by_brand = df.groupby('manufacturer')['price'].mean().reset_index()

fig = px.bar(mean_price_by_brand, x='manufacturer', y='price', labels={'manufacturer':'Brand', 'price':'Average Price'},
             title='Average Price for Each Car Brand', color='manufacturer', color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)



# Calculate the average price per mileage range
mileage_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
mileage_labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+']
df['mileage_range'] = pd.cut(df['odometer'], bins=mileage_bins, labels=mileage_labels)
avg_price_per_range = df.groupby('mileage_range')['price'].mean().reset_index()

# Create a scatter plot
fig = px.scatter(avg_price_per_range, x='mileage_range', y='price', title='Average Price per Mileage Range',
                 labels={'mileage_range': 'Mileage Range', 'price': 'Average Price'})

# Update x-axis to display mileage ranges nicely
fig.update_xaxes(tickvals=mileage_labels, ticktext=[f'{label} ({str(bin)})' for label, bin in zip(mileage_labels, mileage_bins)])

# Show the plot
st.plotly_chart(fig)
