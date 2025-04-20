# River Basin Flood Risk Dashboard
# Team 1 Final Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
import streamlit as st
import folium
from streamlit_folium import folium_static
from PIL import Image
import io
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="River Basin Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stAlert {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .status-box-green {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-weight: bold;
    }
    .status-box-yellow {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeeba;
        text-align: center;
        font-weight: bold;
    }
    .status-box-red {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-weight: bold;
    }
    .dashboard-title {
        text-align: center;
        padding: 10px;
        font-size: 2em;
        font-weight: bold;
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='dashboard-title'>River Basin Flood Risk Dashboard</div>", unsafe_allow_html=True)

# Create tabs for dashboard organization
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Data Analysis", "Historical Trends", "Documentation"])

# Data loading function
@st.cache_data
def load_data():
    # Sample paths - update these with your actual data paths
    rainfall_path = "data/rainfall_data.csv"
    discharge_path = "data/river_discharge.csv"
    water_level_path = "data/water_level.csv"
    
    # Check if files exist, if not use sample data
    if not os.path.exists(rainfall_path):
        # Create synthetic data for demonstration
        # Rainfall data
        dates = pd.date_range(start='2020-01-01', end='2025-04-20', freq='D')
        rainfall = np.random.normal(50, 25, size=len(dates))
        # Add seasonality
        rainfall = rainfall + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
        # Add some extreme events
        rainfall[1500:1510] += 150  # Add a flood event
        rainfall[1200:1205] += 100  # Add another flood event
        
        rainfall_df = pd.DataFrame({
            'date': dates,
            'rainfall_mm': rainfall,
            'station_id': np.random.choice(['ST001', 'ST002', 'ST003', 'ST004', 'ST005'], size=len(dates)),
            'latitude': np.random.uniform(20.5, 21.5, size=len(dates)),
            'longitude': np.random.uniform(85.5, 86.5, size=len(dates))
        })
        
        # Generate historical average
        historical_avg = rainfall_df.groupby(rainfall_df['date'].dt.dayofyear)['rainfall_mm'].mean().reset_index()
        historical_avg.columns = ['dayofyear', 'historical_avg_mm']
        
        # Merge historical average
        rainfall_df['dayofyear'] = rainfall_df['date'].dt.dayofyear
        rainfall_df = pd.merge(rainfall_df, historical_avg, on='dayofyear', how='left')
        
        # Discharge data
        discharge = np.random.normal(200, 75, size=len(dates))
        # Add correlation with rainfall (lagged)
        for i in range(3, len(discharge)):
            discharge[i] += 0.5 * rainfall[i-3]
        
        discharge_df = pd.DataFrame({
            'date': dates,
            'discharge_cumec': discharge,
            'station_id': np.random.choice(['RG001', 'RG002', 'RG003'], size=len(dates)),
            'latitude': np.random.uniform(20.7, 21.2, size=len(dates)),
            'longitude': np.random.uniform(85.8, 86.3, size=len(dates))
        })
        
        # Water level data
        water_level = np.random.normal(3, 1, size=len(dates))
        # Add correlation with discharge
        for i in range(len(water_level)):
            water_level[i] += discharge[i] * 0.01
        
        water_level_df = pd.DataFrame({
            'date': dates,
            'water_level_m': water_level,
            'danger_level_m': np.random.uniform(7, 8, size=len(dates)),
            'warning_level_m': np.random.uniform(5, 6, size=len(dates)),
            'station_id': np.random.choice(['WL001', 'WL002', 'WL003'], size=len(dates)),
            'latitude': np.random.uniform(20.7, 21.2, size=len(dates)),
            'longitude': np.random.uniform(85.8, 86.3, size=len(dates))
        })
        
        # Get unique stations
        rainfall_stations = rainfall_df['station_id'].unique()
        discharge_stations = discharge_df['station_id'].unique()
        waterlevel_stations = water_level_df['station_id'].unique()
        
        # Define stations metadata for mapping
        stations_metadata = pd.DataFrame({
            'station_id': list(rainfall_stations) + list(discharge_stations) + list(waterlevel_stations),
            'station_name': [f"Rainfall Station {i+1}" for i in range(len(rainfall_stations))] + 
                           [f"Discharge Station {i+1}" for i in range(len(discharge_stations))] + 
                           [f"Water Level Station {i+1}" for i in range(len(waterlevel_stations))],
            'station_type': ['Rainfall']*len(rainfall_stations) + ['Discharge']*len(discharge_stations) + ['Water Level']*len(waterlevel_stations),
            'latitude': np.random.uniform(20.5, 21.5, size=len(rainfall_stations) + len(discharge_stations) + len(waterlevel_stations)),
            'longitude': np.random.uniform(85.5, 86.5, size=len(rainfall_stations) + len(discharge_stations) + len(waterlevel_stations))
        })
    else:
        # Load actual data if available
        rainfall_df = pd.read_csv(rainfall_path)
        discharge_df = pd.read_csv(discharge_path)
        water_level_df = pd.read_csv(water_level_path)
        stations_metadata = pd.read_csv("data/stations_metadata.csv")
        
        # Convert date columns
        rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
        discharge_df['date'] = pd.to_datetime(discharge_df['date'])
        water_level_df['date'] = pd.to_datetime(water_level_df['date'])
    
    return rainfall_df, discharge_df, water_level_df, stations_metadata

# Load the data
rainfall_df, discharge_df, water_level_df, stations_metadata = load_data()

# Function to compute average rainfall over a period
def compute_avg_rainfall(df, start_date, end_date, station=None):
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    
    if station:
        mask &= (df['station_id'] == station)
    
    return df.loc[mask, 'rainfall_mm'].mean()

# Function to detect abnormal water levels
def detect_abnormal_levels(df, threshold_factor=1.5):
    # Compute station-wise means and standard deviations
    station_stats = df.groupby('station_id')['water_level_m'].agg(['mean', 'std']).reset_index()
    
    # Merge with original dataframe
    df = pd.merge(df, station_stats, on='station_id')
    
    # Flag abnormal levels (Z-score approach)
    df['z_score'] = (df['water_level_m'] - df['mean']) / df['std']
    df['is_abnormal'] = abs(df['z_score']) > threshold_factor
    
    # Compare with danger and warning levels
    df['status'] = 'Normal'
    df.loc[df['water_level_m'] >= df['warning_level_m'], 'status'] = 'Warning'
    df.loc[df['water_level_m'] >= df['danger_level_m'], 'status'] = 'Danger'
    
    return df

# Detect abnormal water levels
water_level_analysis = detect_abnormal_levels(water_level_df)

# Function to determine overall flood risk
def calculate_flood_risk(rainfall_recent, water_level_status, discharge_recent):
    # Logic to determine flood risk based on multiple factors
    risk_score = 0
    
    # Rainfall contribution (30%)
    if rainfall_recent > 100:
        risk_score += 30
    elif rainfall_recent > 50:
        risk_score += 15
    else:
        risk_score += 0
    
    # Water level contribution (40%)
    if 'Danger' in water_level_status.values:
        risk_score += 40
    elif 'Warning' in water_level_status.values:
        risk_score += 20
    else:
        risk_score += 0
    
    # Discharge contribution (30%)
    max_discharge = 500  # Example threshold
    discharge_percent = min(100, (discharge_recent / max_discharge) * 100)
    risk_score += (discharge_percent / 100) * 30
    
    return risk_score

# Recent data for calculations
recent_date = pd.to_datetime('2025-04-15')
week_ago = recent_date - pd.Timedelta(days=7)

# Get recent rainfall
recent_rainfall = rainfall_df[(rainfall_df['date'] >= week_ago) & (rainfall_df['date'] <= recent_date)]
avg_recent_rainfall = recent_rainfall['rainfall_mm'].mean()

# Get recent water levels
recent_water_levels = water_level_analysis[(water_level_analysis['date'] >= week_ago) & (water_level_analysis['date'] <= recent_date)]
water_level_status = recent_water_levels['status'].value_counts()

# Get recent discharge
recent_discharge = discharge_df[(discharge_df['date'] >= week_ago) & (discharge_df['date'] <= recent_date)]
avg_recent_discharge = recent_discharge['discharge_cumec'].mean()

# Calculate overall flood risk
risk_score = calculate_flood_risk(avg_recent_rainfall, water_level_status, avg_recent_discharge)

# Function to create an interactive map
def create_map(stations_df):
    # Create a map centered at the mean coordinates
    center_lat = stations_df['latitude'].mean()
    center_lon = stations_df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, 
                  tiles="CartoDB positron")
    
    # Add station markers with appropriate colors based on type
    for idx, row in stations_df.iterrows():
        if row['station_type'] == 'Rainfall':
            icon_color = 'blue'
            prefix = 'fa'
            icon = 'cloud'
        elif row['station_type'] == 'Discharge':
            icon_color = 'green'
            prefix = 'fa'
            icon = 'tint'
        else:  # Water Level
            icon_color = 'red'
            prefix = 'fa'
            icon = 'signal'
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"<strong>{row['station_name']}</strong><br>Type: {row['station_type']}<br>ID: {row['station_id']}",
            icon=folium.Icon(color=icon_color, prefix=prefix, icon=icon)
        ).add_to(m)
    
    # Add a river line (simplified example)
    points = stations_df[['latitude', 'longitude']].values
    points = sorted(points, key=lambda p: p[0] + p[1])  # Naive sorting for demo
    
    folium.PolyLine(
        points,
        color='blue',
        weight=3,
        opacity=0.7,
        tooltip='River Path'
    ).add_to(m)
    
    return m

# Generate map
station_map = create_map(stations_metadata)

# Function to create rainfall vs discharge plot
def plot_rainfall_vs_discharge(rainfall_df, discharge_df, days=30):
    end_date = pd.to_datetime('2025-04-20')
    start_date = end_date - pd.Timedelta(days=days)
    
    # Filter data
    rainfall_recent = rainfall_df[(rainfall_df['date'] >= start_date) & (rainfall_df['date'] <= end_date)]
    discharge_recent = discharge_df[(discharge_df['date'] >= start_date) & (discharge_df['date'] <= end_date)]
    
    # Aggregate by date
    rainfall_agg = rainfall_recent.groupby('date')['rainfall_mm'].mean().reset_index()
    discharge_agg = discharge_recent.groupby('date')['discharge_cumec'].mean().reset_index()
    
    # Merge data
    combined = pd.merge(rainfall_agg, discharge_agg, on='date', how='outer')
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add rainfall bars
    fig.add_trace(
        go.Bar(
            x=combined['date'],
            y=combined['rainfall_mm'],
            name="Rainfall (mm)",
            marker_color='blue',
            opacity=0.7
        ),
        secondary_y=False,
    )
    
    # Add discharge line
    fig.add_trace(
        go.Scatter(
            x=combined['date'],
            y=combined['discharge_cumec'],
            name="Discharge (cumec)",
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(width=3, color='red')
        ),
        secondary_y=True,
    )
    
    # Add layout details
    fig.update_layout(
        title_text="Rainfall vs River Discharge",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=False)
    fig.update_yaxes(title_text="Discharge (cumec)", secondary_y=True)
    
    return fig

# Generate rainfall vs discharge plot
rainfall_discharge_plot = plot_rainfall_vs_discharge(rainfall_df, discharge_df)

# Function to create water level trend plot
def plot_water_level_trend(water_level_df, days=30):
    end_date = pd.to_datetime('2025-04-20')
    start_date = end_date - pd.Timedelta(days=days)
    
    # Filter recent data
    recent_data = water_level_df[(water_level_df['date'] >= start_date) & (water_level_df['date'] <= end_date)]
    
    # Aggregate by date and station
    station_agg = recent_data.groupby(['date', 'station_id']).agg({
        'water_level_m': 'mean',
        'danger_level_m': 'first',
        'warning_level_m': 'first'
    }).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add water level lines for each station
    for station in station_agg['station_id'].unique():
        station_data = station_agg[station_agg['station_id'] == station]
        
        fig.add_trace(
            go.Scatter(
                x=station_data['date'],
                y=station_data['water_level_m'],
                name=f"Station {station}",
                mode='lines+markers'
            )
        )
        
        # Add danger level
        danger_level = station_data['danger_level_m'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[station_data['date'].min(), station_data['date'].max()],
                y=[danger_level, danger_level],
                name=f"{station} Danger Level",
                mode='lines',
                line=dict(dash='dash', color='red', width=2),
                showlegend=False
            )
        )
        
        # Add warning level
        warning_level = station_data['warning_level_m'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[station_data['date'].min(), station_data['date'].max()],
                y=[warning_level, warning_level],
                name=f"{station} Warning Level",
                mode='lines',
                line=dict(dash='dot', color='orange', width=2),
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Water Level Trends vs Warning/Danger Levels",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

# Generate water level trend plot
water_level_plot = plot_water_level_trend(water_level_df)

# Function to create rainfall comparison with historical average
def plot_rainfall_historical_comparison(rainfall_df, days=365):
    end_date = pd.to_datetime('2025-04-20')
    start_date = end_date - pd.Timedelta(days=days)
    
    # Filter data
    recent_data = rainfall_df[(rainfall_df['date'] >= start_date) & (rainfall_df['date'] <= end_date)]
    
    # Create monthly aggregates
    recent_data['month'] = recent_data['date'].dt.month
    recent_data['year'] = recent_data['date'].dt.year
    
    monthly_data = recent_data.groupby(['year', 'month']).agg({
        'rainfall_mm': 'sum',
        'historical_avg_mm': lambda x: x.mean() * pd.DatetimeIndex(recent_data['date']).days_in_month.mean()
    }).reset_index()
    
    # Create date strings for x-axis
    monthly_data['date_str'] = monthly_data.apply(lambda x: f"{x['year']}-{x['month']:02d}", axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Add current rainfall bars
    fig.add_trace(
        go.Bar(
            x=monthly_data['date_str'],
            y=monthly_data['rainfall_mm'],
            name="Current Rainfall",
            marker_color='blue'
        )
    )
    
    # Add historical average line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['date_str'],
            y=monthly_data['historical_avg_mm'],
            name="Historical Average",
            mode='lines+markers',
            line=dict(dash='dash', color='gray', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Monthly Rainfall vs Historical Average",
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

# Generate rainfall historical comparison plot
rainfall_historical_plot = plot_rainfall_historical_comparison(rainfall_df)

# Function to generate PDF report
def generate_report():
    # This would typically use a library like ReportLab or pdfkit
    # For this example, we'll just prepare the content that would go into the report
    
    report_content = {
        'title': 'River Basin Flood Risk Assessment Report',
        'date': pd.to_datetime('today').strftime('%Y-%m-%d'),
        'summary': {
            'risk_score': risk_score,
            'risk_level': 'High' if risk_score > 70 else 'Medium' if risk_score > 40 else 'Low',
            'avg_rainfall': avg_recent_rainfall,
            'avg_discharge': avg_recent_discharge,
            'abnormal_stations': water_level_analysis[water_level_analysis['is_abnormal']].shape[0]
        },
        'recommendations': [
            'Continue monitoring water levels at stations showing abnormal readings',
            'Prepare emergency response teams if rainfall continues at current rate',
            'Notify local authorities of potential flood risk'
        ] if risk_score > 40 else [
            'Regular monitoring recommended',
            'No immediate action required'
        ]
    }
    
    return report_content

# Generate report content
report_content = generate_report()

# Dashboard Tab Content
with tab1:
    st.subheader("Current Flood Risk Status")
    
    # Risk indicator
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    
    with risk_col1:
        if risk_score > 70:
            st.markdown(f"<div class='status-box-red'>HIGH RISK<br>{risk_score:.1f}%</div>", unsafe_allow_html=True)
        elif risk_score > 40:
            st.markdown(f"<div class='status-box-yellow'>MEDIUM RISK<br>{risk_score:.1f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-box-green'>LOW RISK<br>{risk_score:.1f}%</div>", unsafe_allow_html=True)
    
    with risk_col2:
        # Recent rainfall status
        rainfall_percent = min(100, (avg_recent_rainfall / 100) * 100)
        st.metric(
            label="Recent Rainfall",
            value=f"{avg_recent_rainfall:.1f} mm",
            delta=f"{rainfall_percent:.1f}% of threshold"
        )
    
    with risk_col3:
        # Recent discharge status
        discharge_percent = min(100, (avg_recent_discharge / 500) * 100)
        st.metric(
            label="Recent Discharge",
            value=f"{avg_recent_discharge:.1f} cumec",
            delta=f"{discharge_percent:.1f}% of threshold"
        )
    
    st.markdown("---")
    
    # Map and recent trends
    map_col, trends_col = st.columns([1, 1])
    
    with map_col:
        st.subheader("River Basin Monitoring Stations")
        folium_static(station_map)
    
    with trends_col:
        st.subheader("Water Level Status")
        st.plotly_chart(water_level_plot, use_container_width=True)
    
    # Rainfall and discharge plots
    st.subheader("Rainfall vs Discharge Relationship")
    st.plotly_chart(rainfall_discharge_plot, use_container_width=True)
    
    # Abnormal readings alert
    abnormal_stations = water_level_analysis[water_level_analysis['is_abnormal'] & 
                                           (water_level_analysis['date'] >= week_ago)]
    
    if not abnormal_stations.empty:
        st.markdown("### ⚠️ Abnormal Water Level Readings Detected")
        st.dataframe(abnormal_stations[['date', 'station_id', 'water_level_m', 'z_score', 'status']])

# Data Analysis Tab Content
with tab2:
    st.header("Data Analysis Tools")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        st.subheader("Station Selection")
        
        # Station type selector
        station_type = st.selectbox(
            "Select station type",
            ["Rainfall", "Discharge", "Water Level"]
        )
        
        # Station selector based on type
        filtered_stations = stations_metadata[stations_metadata['station_type'] == station_type]
        selected_station = st.selectbox(
            "Select station",
            filtered_stations['station_id'].tolist(),
            format_func=lambda x: f"{x} - {filtered_stations[filtered_stations['station_id'] == x]['station_name'].iloc[0]}"
        )
    
    with analysis_col2:
        st.subheader("Time Period")
        
        # Date range selector
        date_range = st.date_input(
            "Select date range",
            value=(pd.to_datetime('2025-03-01').date(), pd.to_datetime('2025-04-20').date()),
            max_value=pd.to_datetime('2025-04-20').date()
        )
    
    # Display data based on selection
    if station_type == "Rainfall":
        # Filter data for selected station and date range
        mask = ((rainfall_df['station_id'] == selected_station) & 
                (rainfall_df['date'] >= pd.to_datetime(date_range[0])) & 
                (rainfall_df['date'] <= pd.to_datetime(date_range[1])))
        
        filtered_data = rainfall_df[mask]
        
        if not filtered_data.empty:
            st.subheader(f"Rainfall Data for Station {selected_station}")
            
            # Summary statistics
            avg_rainfall = filtered_data['rainfall_mm'].mean()
            max_rainfall = filtered_data['rainfall_mm'].max()
            total_rainfall = filtered_data['rainfall_mm'].sum()
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Average Daily Rainfall", f"{avg_rainfall:.2f} mm")
            stat_col2.metric("Maximum Daily Rainfall", f"{max_rainfall:.2f} mm")
            stat_col3.metric("Total Rainfall", f"{total_rainfall:.2f} mm")
            
            # Plot rainfall data
            fig = px.line(
                filtered_data, 
                x='date', 
                y='rainfall_mm',
                title=f"Daily Rainfall at Station {selected_station}",
                labels={'rainfall_mm': 'Rainfall (mm)', 'date': 'Date'}
            )
            fig.add_trace(
                go.Scatter(
                    x=filtered_data['date'],
                    y=filtered_data['historical_avg_mm'],
                    mode='lines',
                    name='Historical Average',
                    line=dict(dash='dash', color='gray')
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw data
            with st.expander("Show Raw Data"):
                st.dataframe(filtered_data)
            
    elif station_type == "Discharge":
        # Filter data for selected station and date range
        mask = ((discharge_df['station_id'] == selected_station) & 
                (discharge_df['date'] >= pd.to_datetime(date_range[0])) & 
                (discharge_df['date'] <= pd.to_datetime(date_range[1])))
        
        filtered_data = discharge_df[mask]
        
        if not filtered_data.empty:
            st.subheader(f"Discharge Data for Station {selected_station}")
            
            # Summary statistics
            avg_discharge = filtered_data['discharge_cumec'].mean()
            max_discharge = filtered_data['discharge_cumec'].max()
            
            stat_col1, stat_col2 = st.columns(2)
            stat_col1.metric("Average Discharge", f"{avg_discharge:.2f} cumec")
            stat_col2.metric("Maximum Discharge", f"{max_discharge:.2f} cumec")
            
            # Plot discharge data
            fig = px.line(
                filtered_data, 
                x='date', 
                y='discharge_cumec',
                title=f"Daily Discharge at Station {selected_station}",
                labels={'discharge_cumec': 'Discharge (cumec)', 'date': 'Date'}
            )
            
            # Add a horizontal line representing a theoretical threshold
            fig.add_hline(y=400, line_dash="dash", line_color="red", 
                         annotation_text="Warning Threshold", 
                         annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display raw data
            with st.expander("Show Raw Data"):
                st.dataframe(filtered_data)
    
    else:  # Water Level
        # Filter data for selected station and date range
        mask = ((water_level_df['station_id'] == selected_station) & 
                (water_level_df['date'] >= pd.to_datetime(date_range[0])) & 
                (water_level_df['date'] <= pd.to_datetime(date_range[1])))
        
        filtered_data = water_level_df[mask]
        
        if not filtered_data.empty:
            st.subheader(f"Water Level Data for Station {selected_station}")
            
            # Summary statistics
            avg_level = filtered_data['water_level_m'].mean()
            max_level = filtered_data['water_level_m'].max()
            danger_level = filtered_data['danger_level_m'].iloc[0]
            warning_level = filtered_data['warning_level_m'].iloc[0]
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric("Average Level", f"{avg_level:.2f} m")
            stat_col2.metric("Maximum Level", f"{max_level:.2f} m")
            stat_col3.metric("Warning Level", f"{warning_level:.2f} m")