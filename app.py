import streamlit as st
import pandas as pd
import plotly_express as px
import matplotlib as plt
import numpy as np

df = pd.read_csv("vehicles_us.csv")
df['manufacturer'] = df['model'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)

# Define a function to fill missing values with the mode of each group
def fill_missing_with_group_mode(series):
    return series.fillna(series.mode().iloc[0])

# Group by 'model' and fill missing values in 'model_year' column with group mode
df['model_year'] = df.groupby(['model'])['model_year'].transform(fill_missing_with_group_mode)


# Define a function to fill missing values with the median of each group
def fill_missing_with_group_median(series):
    return series.fillna(series.median())

# Group by 'model' and 'model_year' and fill missing values in 'cylinders' column with group median
df['cylinders'] = df.groupby(['model', 'model_year'])['cylinders'].transform(fill_missing_with_group_median)


# Drop odometer, paint_color and is_4wd columns
df = df.drop(['odometer', 'paint_color','is_4wd'], axis=1)

# Drop missing values
df= df.dropna()

st.header("Final project sprint 4 \n")
st.header("Cars project dataset")
st.dataframe(df)

#Plot vehicles types per manufacturer
st.header("Vehicles types by manufacturer")
fig = px.histogram(df, x='manufacturer', color='type')
fig.update_layout(yaxis_title='Number of Vehicles')
st.write(fig)

st.header('Conditions vs model year')
fig2 = px.histogram(df, x='model_year', color='condition')
fig2.update_layout(yaxis_title='Number of Vehicles')
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
        title = f"Car models selected, models number chosen: {len(selected_models)}"
    else:
        filtered_df = df
        title = 'All Car Models'
    
    model_counts = filtered_df['model'].value_counts()
    fig = px.bar(x=model_counts.index, y=model_counts.values, labels={'x':'Model', 'y':'Count'},
                 title=title, color_discrete_sequence=['#007F73'])
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(yaxis_title='Number of Vehicles')
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



# Relation between model year and price
# Filter the DataFrame to include only price values <= 100000
df_filtered = df[df['price'] <= 100000]

# Create the scatter plot
fig = px.scatter(df_filtered, x='model_year', y='price', title='Model Year vs Price', 
                 labels={'model_year': 'Model Year', 'price': 'Price'},
                 template='plotly_white', opacity=0.5, color_discrete_sequence=["#007F73"])

# Update the y-axis range and tick interval
fig.update_yaxes(range=[0, 100000], tickvals=list(range(0, 100001, 10000)))

# Display the plot
st.plotly_chart(fig)
