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

brand_counts = df['manufacturer'].value_counts()
top_brands = brand_counts.head(20)

fig = px.bar(x=top_brands.index, y=top_brands.values, labels={'x':'Brand', 'y':'Count'},
             title='Top 20 Most Popular Car Brands', color_discrete_sequence=['#007F73'])
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

#Car model popularity
model_counts = df['model'].value_counts()
top_models = model_counts.head(20)

fig = px.bar(x=top_models.index, y=top_models.values, labels={'x':'Model', 'y':'Count'},
             title='Top 20 Most Popular Car Models', color_discrete_sequence=['#007F73'])
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

#Mean brand price

mean_price_by_brand = df.groupby('manufacturer')['price'].mean().reset_index()

fig = px.bar(mean_price_by_brand, x='manufacturer', y='price', labels={'manufacturer':'Brand', 'price':'Average Price'},
             title='Average Price for Each Car Brand', color='manufacturer', color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)


# Avg days listed by brand
average_days_listed_by_brand = df.groupby('manufacturer')['days_listed'].mean().reset_index()

fig = px.bar(average_days_listed_by_brand, x='manufacturer', y='days_listed', labels={'manufacturer':'Brand', 'days_listed':'Average Days Listed'},
             title='Average Days Listed for Each Car Brand', color='manufacturer', color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig)

# Price ranges
price_bins = np.linspace(df['price'].min(), df['price'].max(), num=6)
price_labels = [f"${int(price_bins[i])}-{int(price_bins[i+1])}" for i in range(len(price_bins)-1)]

# New column for price range
df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, include_lowest=True)

# Calculate average days listed per price range
average_days_listed_by_price_range = df.groupby('price_range')['days_listed'].mean().reset_index()

# Define the price ranges
price_ranges = [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000), (40000, 50000), (50000, float('inf'))]

# Calculate the average days listed for each price range
avg_days_listed_per_range = []
for price_range in price_ranges:
    lower_bound, upper_bound = price_range
    if upper_bound == float('inf'):
        filtered_df = df[(df['price'] >= lower_bound)]
    else:
        filtered_df = df[(df['price'] >= lower_bound) & (df['price'] < upper_bound)]
    avg_days_listed = filtered_df['days_listed'].mean()
    avg_days_listed_per_range.append(avg_days_listed)

# Create a DataFrame for the plot
plot_data = {'Price Range': ['${}-{}'.format(pr[0], pr[1]) if pr[1] != float('inf') else '${}+'.format(pr[0]) for pr in price_ranges],
             'Average Days Listed': avg_days_listed_per_range}
plot_df = pd.DataFrame(plot_data)

# Create a bar chart using Plotly Express
fig = px.bar(plot_df, x='Price Range', y='Average Days Listed', color='Price Range',
             title='Average Days Listed per Price Range')
fig.update_xaxes(tickangle=45)
fig.show()

# Calculate the average days listed per car type
avg_days_listed_per_type = df.groupby('type')['days_listed'].mean().reset_index()

# Create a bar chart using Plotly Express
fig = px.bar(avg_days_listed_per_type, x='type', y='days_listed',
             title='Average Days Listed per Car Type',
             labels={'type': 'Car Type', 'days_listed': 'Average Days Listed'})
fig.update_xaxes(title_text=None, tickangle=45)
fig.show()

df = pd.DataFrame({
    'odometer': [25000, 75000, 125000, 175000, 225000],
    'price': [15000, 12000, 9000, 6000, 4000]
})

# Define mileage ranges
mileage_bins = [0, 50000, 100000, 150000, 200000, float('inf')]
mileage_labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k+']

# Create a new column with mileage ranges
df['mileage_range'] = pd.cut(df['odometer'], bins=mileage_bins, labels=mileage_labels)

# Calculate the average price per mileage range
avg_price_per_mileage_range = df.groupby('mileage_range')['price'].mean().reset_index()

# Plot the average price per mileage range using Plotly Express
fig = px.bar(avg_price_per_mileage_range, x='mileage_range', y='price', 
             title='Average Price per Mileage Range',
             labels={'mileage_range': 'Mileage Range', 'price': 'Average Price'})
fig.show()