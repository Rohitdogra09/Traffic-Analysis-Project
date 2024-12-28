# Traffic Pattern Analysis and Optimization Project

# Import Required Libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import folium
from sklearn.cluster import KMeans
from statsmodels.tsa.arima_model import ARIMA

# Step 1: Data Collection
# Placeholder for loading traffic dataset
def load_data(file_path):
    """Load traffic data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

# Example dataset structure: ['timestamp', 'latitude', 'longitude', 'vehicle_count', 'average_speed', 'accidents']
data = load_data("traffic_data.csv")

# Step 2: Data Cleaning
# Handle missing values
data.dropna(inplace=True)

# Convert timestamp to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Step 3: Exploratory Data Analysis (EDA)
# Plot traffic volume trends
def plot_traffic_trends(data):
    data['hour'] = data['timestamp'].dt.hour
    hourly_traffic = data.groupby('hour')['vehicle_count'].mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values)
    plt.title('Average Traffic Volume by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Vehicle Count')
    plt.show()

plot_traffic_trends(data)

# Step 4: Geospatial Analysis
# Visualize accident-prone areas on a map
def plot_accident_map(data):
    accident_map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)
    for _, row in data.iterrows():
        if row['accidents'] > 0:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='red',
                fill=True,
                fill_opacity=0.6
            ).add_to(accident_map)
    return accident_map

accident_map = plot_accident_map(data)
accident_map.save("accident_map.html")

# Step 5: Clustering (Identify Bottleneck Areas)
# Prepare data for clustering
geo_data = data[['latitude', 'longitude']]

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(geo_data)

# Visualize clusters on the map
def plot_cluster_map(data):
    cluster_map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=colors[row['cluster']],
            fill=True,
            fill_opacity=0.6
        ).add_to(cluster_map)
    return cluster_map

cluster_map = plot_cluster_map(data)
cluster_map.save("cluster_map.html")

# Step 6: Time-Series Analysis for Traffic Prediction
# Aggregate vehicle count by hour
time_series_data = data.groupby(data['timestamp'].dt.hour)['vehicle_count'].sum()

# Fit ARIMA Model
model = ARIMA(time_series_data, order=(1, 1, 1))
model_fit = model.fit(disp=0)

# Forecast Traffic Volume
forecast = model_fit.forecast(steps=10)[0]
print("Traffic Forecast for Next 10 Hours:", forecast)

# Step 7: Reporting
# Save insights to a report
def save_report(data, forecast):
    with open("traffic_analysis_report.txt", "w") as file:
        file.write("Traffic Analysis Report\n")
        file.write("======================\n")
        file.write("\nClusters Identified:\n")
        file.write(str(data['cluster'].value_counts()))
        file.write("\n\nTraffic Forecast for Next 10 Hours:\n")
        file.write(str(forecast))

save_report(data, forecast)

# End of Project
