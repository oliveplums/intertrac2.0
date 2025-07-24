import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from datetime import timedelta
from typing import Sequence
import urllib3
import sqlite3
import geopandas as gpd
import plotly.express as px
import os 
from shapely.geometry import Point
import plotly.graph_objects as go
import numpy as np
import math

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- API fetch functions ----
def get_voyage_history(username: str, password: str, imo: Sequence[int], from_date: datetime, to_date: datetime):
    imo_str = ','.join(map(str, imo))
    url = f"https://position.stratumfive.com/ais/staticvoyage-by-imo/{imo_str}/{from_date.isoformat()}/{to_date.isoformat()}"
    response = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password), verify=False)
    response.raise_for_status()
    return response.json()

def get_ais_positions(username: str, password: str, mmsis: Sequence[int], from_date: datetime, to_date: datetime, sixhourly: str = 'true'):
    mmsi_str = ','.join(map(str, mmsis))
    url = f"https://position.stratumfive.com/ais/positions/{mmsi_str}/{from_date.isoformat()}/{to_date.isoformat()}?is6hourly={sixhourly}"
    response = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password), verify=False)
    response.raise_for_status()
    return response.json()

def detect_mmsi_changes(voyage_data, end_date: str):
    mmsi_values = [entry['mmsi'] for entry in voyage_data]
    mmsis = list(set(mmsi_values))
    previous_mmsi = None
    timestamp_changes = []

    for entry in voyage_data:
        current_mmsi = entry['mmsi']
        timestamp = entry['timestamp']
        if current_mmsi != previous_mmsi:
            if previous_mmsi is not None:
                timestamp_changes.append((previous_mmsi, current_mmsi, timestamp))
        previous_mmsi = current_mmsi

    if not timestamp_changes:
        timestamp_changes.append((mmsis[0], mmsis[0], end_date))
    elif len(timestamp_changes) > 1:
        source = timestamp_changes[-1][1]
        sourcetime = timestamp_changes[-1][2]
        for i in range(len(timestamp_changes) - 1, -1, -1):
            if timestamp_changes[i][0] in [row[1] for row in timestamp_changes[:i]]:
                timestamp_changes.pop(i)
        timestamp_changes = [(timestamp_changes[0][0], source, sourcetime)]

    return timestamp_changes

def fetch_and_combine_ais(username, password, timestamp_changes, start, end, sixhourly):
    df_combined = pd.DataFrame()
    start_dt = datetime.fromisoformat(start + 'T00:00:00')
    end_dt = datetime.fromisoformat(end + 'T00:00:00')

    mmsi_list = [timestamp_changes[0][0], timestamp_changes[0][1]]
    # If both MMSIs are the same, just keep one
    if mmsi_list[0] == mmsi_list[1]:
        mmsi_list = [mmsi_list[0]]

    for mmsi in mmsi_list:
        ais_data = get_ais_positions(username, password, [mmsi], start_dt, end_dt, sixhourly)
        df = pd.DataFrame.from_dict(ais_data)
        if df.empty:
            continue

        df['DateTime'] = pd.to_datetime(df['timestamp'])
        df['latitude'] = df['lat']
        df['longitude'] = df['lon']
        df['speed'] = df['sogKts']
        df = df[['DateTime', 'speed', 'latitude', 'longitude']]

        switch_time = datetime.fromisoformat(timestamp_changes[0][2])
        if mmsi == timestamp_changes[0][0]:
            df = df[df['DateTime'] < switch_time]
        else:
            df = df[df['DateTime'] > switch_time]

        df_combined = pd.concat([df_combined, df])

    df_cleaned = df_combined.drop_duplicates(subset='DateTime').reset_index(drop=True)
    df = df_cleaned[df_cleaned['speed'] < 30]
    df = df.drop_duplicates(subset='DateTime').sort_values('DateTime').reset_index(drop=True)
    return df

correct_ids = pd.DataFrame({
    "LME_NAME": [
        "Agulhas Current", "Aleutian Islands", "Antarctica", "Arabian Sea", "Central Arctic",
        "Baltic Sea", "Barents Sea", "Bay of Bengal", "Beaufort Sea", "Benguela Current",
        "Black Sea", "California Current", "Canary Current", "Caribbean Sea", "Celtic-Biscay Shelf",
        "Northern Bering - Chukchi Seas", "East Bering Sea", "East Brazil Shelf", "East Central Australian Shelf", 
        "East China Sea", "Greenland Sea", "East Siberian Sea", "Faroe Plateau", "Guinea Current",
        "Gulf of Alaska", "Gulf of California", "Gulf of Mexico", "Gulf of Thailand", "Hudson Bay Complex",
        "Humboldt Current", "Iberian Coastal", "Iceland Shelf and Sea", "Indonesian Sea", "Insular Pacific-Hawaiian",
        "Kara Sea", "Kuroshio Current", "Laptev Sea", "Mediterranean Sea", "New Zealand Shelf",
        "Labrador - Newfoundland", "North Australian Shelf", "North Brazil Shelf", "Canadian High Arctic - North Greenland",
        "North Sea", "Northeast Australian Shelf", "Northeast U.S. Continental Shelf", "Northwest Australian Shelf",
        "Norwegian Sea", "Oceanic (open ocean)", "Oyashio Current", "Pacific Central-American Coastal",
        "Patagonian Shelf", "Red Sea", "Scotian Shelf", "Sea of Japan", "Sea of Okhotsk",
        "Somali Coastal Current", "South Brazil Shelf", "South China Sea", "South West Australian Shelf",
        "Southeast Australian Shelf", "Southeast U.S. Continental Shelf", "Sulu-Celebes Sea", "West Bering Sea",
        "West Central Australian Shelf", "Canadian Eastern Arctic - West Greenland", "Yellow Sea"
    ],
    "LME_NUMBER": [
        40, 59, 47, 22, 65, 1, 54, 27, 63, 37,
        11, 10, 17, 25, 6, 66, 60, 35, 42, 18,
        52, 57, 50, 32, 4, 20, 21, 30, 62, 36,
        14, 51, 34, 24, 55, 15, 56, 13, 44, 5,
        38, 31, 64, 3, 39, 12, 41, 53, 0, 8,
        28, 45, 23, 9, 7, 2, 33, 43, 26, 48,
        46, 19, 29, 58, 49, 61, 16
    ]
# --- Streamlit UI ---
st.title("🚢 AIS Dashboard")

username = st.secrets["username"]
password = st.secrets["password"]

imo_input = st.text_input("IMO number(s) (comma separated)", value="9770634")
start_date = st.date_input("Start Date", value=datetime(2025, 7, 1))
end_date = st.date_input("End Date", value=datetime(2025, 7, 22))
sixhourly = st.selectbox("6-Hourly data?", options=["true", "false"], index=0)

# Session state handling for the checkbox
if "mmsi_gap_enabled" not in st.session_state:
    st.session_state["mmsi_gap_enabled"] = False

# Now show the checkbox, linked to session state
mmsi_gap_checkbox = st.checkbox("Check MMSI Change: Gap Search", value=st.session_state["mmsi_gap_enabled"])

            
if st.button("Fetch Data"):
    with st.spinner("Fetching voyage and AIS data..."):
        try:
            imo_list = list(map(int, imo_input.split(',')))
                
            # Always get voyage_data
            if mmsi_gap_checkbox==True:
                voyage_data = get_voyage_history(username, password, imo_list, start_date, end_date)
                timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())
            else:
                days_ago = end_date- timedelta(days=10)
                voyage_data = get_voyage_history(username, password, imo_list, days_ago, end_date)
                timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())
                
# ---- SQLite vessel info ----
            # try:
            #     imo_for_db = imo_list[0]
            #     with sqlite3.connect("my_sqlite.db") as cnn:
            #         query = f"SELECT * FROM vesselInfo WHERE LRIMOShipNo = {imo_for_db};"
            #         dfVesselInfo = pd.read_sql(query, cnn)

            #     if not dfVesselInfo.empty:
            #         st.subheader("📄 Vessel Information from Local DB")
            #         st.dataframe(dfVesselInfo)
            #     else:
            #         st.warning(f"No vessel info found in local DB for IMO {imo_for_db}")
            # except Exception as e:
            #     st.error(f"SQLite DB error: {e}")
        
            if not timestamp_changes or len(timestamp_changes[0]) < 3:
                st.error("Unable to detect MMSI transitions – voyage data might be incomplete or invalid.")
            else:
                df_ais = fetch_and_combine_ais(
                    username, password,
                    timestamp_changes,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    sixhourly
                )
        
                if df_ais.empty:
                    st.warning("No AIS position data found for the selected criteria.")
                else:
                    st.success("AIS Data fetched successfully!")



                # ---- LME Shapefile and Excel Info ----
                try:
                    LMEPolygon = "LMEPolygon1/LMEs66.shp"
                    LMEPolygon_path = os.path.abspath(LMEPolygon)
                    LME_sf = gpd.read_file(LMEPolygon_path)
                    LEM_gsd_new = LME_sf.to_crs(epsg=4326)

                    LEM_gsd_new.dropna(subset=['OBJECTID'], inplace=True)
                    LEM_gsd_new.drop(columns=['LME_NUMBER'], inplace=True)
                    LEM_gsd_new = LEM_gsd_new.merge(correct_ids, on="LME_NAME", how="left")

                    LME = pd.read_excel("LME values.xlsx")
                    LME.columns = LME.iloc[0]
                    LME = LME[1:].reset_index(drop=True)

                    AIS_long_lat = df_ais[['longitude', 'latitude']]
                    AIS_long_lat.columns = ['Longitude', 'Latitude']
                    points_cords = [Point(xy) for xy in zip(AIS_long_lat.Longitude, AIS_long_lat.Latitude)]
                    Route = gpd.GeoDataFrame(AIS_long_lat, geometry=points_cords, crs='EPSG:4326')

                    Route = gpd.sjoin(Route, LEM_gsd_new[['geometry', 'LME_NUMBER']], how="left", predicate='within')
                    Route['ID'] = Route['LME_NUMBER']
                    Route['Datetime'] = df_ais['DateTime']
                    result = pd.merge(Route, LME, how="left", on="ID")
                    result['months'] = result['Datetime'].apply(lambda x: x.strftime('%b'))

                    # Risk calculation
                    b = [0] * len(Route['geometry'])
                    Winter = ['Nov', 'Dec', 'Jan']
                    Spring = ['Feb', 'Mar', 'Apr']
                    Summer = ['May', 'Jun', 'Jul']
                    Autumn = ['Aug', 'Sep', 'Oct']

                    for i in range(len(result['geometry'])):
                        if result['months'][i] in Winter:
                            b[i] = result['Nov - Jan'][i]
                        elif result['months'][i] in Spring:
                            b[i] = result['Feb - Apr'][i]
                        elif result['months'][i] in Summer:
                            b[i] = result['May - Jul'][i]
                        else:
                            b[i] = result['Aug - Oct'][i]

                    df_ais['risk'] = b

                except Exception as e:
                    st.error(f"Geospatial or Excel error: {e}")
        except Exception as e:
            st.error("Unexpected error during API call.")

            
##############Speed and Activity Summary#######################
        st.set_page_config(layout="wide")
        # Time difference between points
        
        #df_ais = df_ais[df_ais["speed"] < 30]
        
        df_ais['Diff'] = df_ais['DateTime'].diff().fillna(pd.Timedelta(0))

        
        # Distance in nautical miles (speed in knots * hours)
        df_ais['distance'] = df_ais['speed'] * 0.0002777778 * df_ais['Diff'].dt.total_seconds()
        
        # Sea Miles per Month (SMM)
        n = len(df_ais) - 1
        total_time_months = (df_ais['DateTime'].iloc[n] - df_ais['DateTime'].iloc[0]).total_seconds() / 2.628e+6
        smm = df_ais['distance'].sum() / total_time_months if total_time_months else 0
        
        # Total sea miles
        total_miles = df_ais['distance'].sum()
        
        # % Activity above 10 knots
        above_10 = df_ais[df_ais['speed'] > 10]
        perc_above_10 = (above_10['Diff'].sum() / df_ais['Diff'].sum()) * 100 if df_ais['Diff'].sum().total_seconds() > 0 else 0
        
        # % Activity above 3 knots
        above_3 = df_ais[df_ais['speed'] > 3]
        perc_above_3 = (above_3['Diff'].sum() / df_ais['Diff'].sum()) * 100 if df_ais['Diff'].sum().total_seconds() > 0 else 0
        

        # Create the summary table
        summary_df = pd.DataFrame({
            "Metric": ["Sea Miles per Month (SMM)", "Total Sea Miles", "% Activity above 10 knots", "% Activity above 3 knots"],
            f"{imo_list[0]}": [round(smm, 0), round(total_miles, 0), f"{perc_above_10:.0f}%", f"{perc_above_3:.0f}%"]
        })

        # Convert Value to string for consistent formatting
        summary_df[f"{imo_list[0]}"] = summary_df[f"{imo_list[0]}"].astype(str)


        # Transpose the dataframe to make metrics the column headers
        summary_transposed = summary_df.set_index("Metric").T

        # Apply styling
        styled_df = summary_transposed.style.set_properties(**{
            'background-color': '#f0f2f6',
            'color': 'black',
            'border-color': 'gray',
            'font-size': '16px',
            'text-align': 'left'
        }).hide(axis="index")

        # Display
        st.subheader("📈 Speed and Activity Summary")
        st.dataframe(styled_df, use_container_width=True)


#####MAP###############

        # Define colors for each risk level
        colours = {'null': 'grey', 'VL': 'darkgreen', 'L': '#37ff30', 'M': 'yellow', 'H': 'orange', 'VH': 'red'}
        
        # Make sure you're using the correct dataframe
        df = df_ais.copy()
        
        # Handle potential missing values in 'risk'
        df['risk'] = df['risk'].fillna('VL')
        
        # Compute the bounds of the data
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        
        # Calculate center of map
        center_lon = df['longitude'].mean()
        center_lat = df['latitude'].mean()
        
        # Define zoom level calculation
        def calculate_zoom_level(lon_range, lat_range):
            max_range = max(lon_range, lat_range)
            zoom = -math.log2(max_range / 360)
            return max(min(zoom, 4), 2)
        
        # Calculate zoom based on geographic range
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        zoom = calculate_zoom_level(lon_range, lat_range)
        
        # Create the Scattermapbox figure
        fig = go.Figure(go.Scattermapbox(
            lon=df['longitude'],
            lat=df['latitude'],
            mode='markers',
            marker={
                'size': 6,
                'color': [colours.get(label, 'grey') for label in df['risk']],
            },
            # Add DateTime to hover text
            text=[
                f"{dt.strftime('%d %B %Y')}"
                for dt in df['DateTime']
            ],
            hoverinfo='text'  # ensures only the 'text' content shows on hover
        ))
        
        
        # Set map layout
        fig.update_layout(
            mapbox={
                'style': "open-street-map",
                'center': {'lon': center_lon, 'lat': center_lat},
                'zoom': zoom,
            },
            showlegend=False,
            width=1000,    # Adjust width as needed
            height=1000    # Same height to make it square
        )

        # Display the map in Streamlit
        st.subheader("🗺️ Vessel Route and Risk Map")
        st.plotly_chart(fig,  key="map")

########### Speed Timeline#########################
        # Convert 'DateTime' column to datetime format
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Sort the DataFrame by 'DateTime'
        df = df.sort_values('DateTime')
        
        # Define the color palette
        akzo_primary = [(51/255, 102/255, 0/255), (0/255, 255/255, 0/255), (255/255, 255/255, 0/255), (255/255, 153/255, 0/255), (255/255, 0/255, 0/255),(0/255, 81/255, 146/255)]
        
        # Define the order of the 'Fouling Challenge' categories
        fouling_challenge_order = ['VL', 'L', 'M', 'H', 'VH']
        
        # Create a dictionary that maps each category to a color
        color_map = dict(zip(fouling_challenge_order, akzo_primary[:5]))
        
        # Filter DataFrame to only include rows where 'Speed' is less than or equal to 30
        df = df[df['speed'] <= 30]
        
        # Initialize figure
        fig = go.Figure()
        
        # Plot each row as a separate line
        for i in range(1, len(df)):
            risk_value = df['risk'].iloc[i]
            
            # Convert RGB tuple to 'rgb(r, g, b)' string
            color = color_map.get(risk_value, (0, 0, 0))  # Default to black if missing
            color = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
            
            fig.add_trace(go.Scatter(
                x=df['DateTime'].iloc[i-1:i+1], 
                y=df['speed'].iloc[i-1:i+1],
                mode='lines',
                line=dict(color=color, width=2),
                name=str(risk_value)
            ))
        
        
        
        # Customize plot title and labels
        fig.update_layout(title='Speed Timeline',
            yaxis_title='Speed / knots',
            xaxis_tickformat='%d %B %Y',template='plotly_white', autosize=False, bargap=0.30,
                          font=dict(color='grey',size=14),
                           showlegend=False,# Set the font color to black and size to 12
                          width=1000,  # Set the width of the figure to make the graph longer
                          height=500   # Optionally set the height of the figure
        )
        st.subheader("Speed Timeline")
        st.plotly_chart(fig, key="speed timeline")

########### Speed Histogram #########################


        # Assuming 'df' is already defined and loaded with the relevant data
        #df['Diff'] = df['Diff'].dt.total_seconds()
        
        # Filter speeds greater than or equal to 5 knots and round them
        df['histspeed'] = round(df['speed'], 5)
        dfabove5 = df[df['histspeed'] >= 5]
        
        # Create the histogram
        histfig = px.histogram(dfabove5, x="histspeed", histnorm='percent',nbins=round(dfabove5['histspeed'].max())+1, 
                           labels={"histspeed": "Speed / Knots", "percent": "Percentage(%)"},
                           title="Speed Histogram", color_discrete_sequence=['#3273a8'])
        
        histfig.update_layout(template='plotly_white', autosize=False, bargap=0.30,
                        yaxis_title="Percentage (%)",
                          font=dict(color='grey',size=14),  # Set the font color to black and size to 12
                          width=400,  # Set the width of the histfigure to make the graph longer
                          height=400   # Optionally set the height of the histfigure
        )


        # Display the chart in half the page
        col1, col2 = st.columns([1, 1])  # Two equal-width columns

        with col1:  # or use col2 if you prefer
            st.subheader("Speed Histogram")
            st.plotly_chart(histfig, use_container_width=True, key="speed hist")


########## Fouling Challenge #############

        # Define x-axis labels
        x = ['null','VL', 'L', 'M', 'H', 'VH']
        
        # Calculate the percentage weight for each risk level
        df['weight'] = (100 * df['Diff'] / df['Diff'].sum())
        
        # Calculate the sum of weights for each risk level
        y = [df.loc[df["risk"] == label, "weight"].sum() for label in x]
        
        # Define colors for each risk level
        colours = {'null':'grey','VL': 'darkgreen', 'L': '#37ff30', 'M': 'yellow', 'H': 'orange', 'VH': 'red'}
        
        # Create the bar plot with custom colors and outlined bars
        fig = go.Figure(data=[go.Bar(x=x, y=y, 
                                     marker_color=[colours[label] for label in x], 
                                     marker_line_color='black',  # Add black outline
                                     marker_line_width=1)])  # Adjust the width of the outline if needed
        
        # Add value labels above each bar
        for i, val in enumerate(y):
            fig.add_annotation(
                x=x[i],
                y=val,
                text=f'{int(val)}',
                showarrow=False,
                font=dict(color='black', size=13),
                yshift=10
            )
        
        # Customize plot title and labels
        fig.update_layout(title="FOULING CHALLENGE", xaxis_title='Fouling Challenge', 
                          template='plotly_white', autosize=False, bargap=0.30,
                          yaxis_title="Percentage (%)",
                          font=dict(color='grey',size=14),  # Set the font color to black and size to 12
                          width=400,  # Set the width of the figure to make the graph longer
                          height=400   # Optionally set the height of the figure
        )
        with col2:
            st.subheader("Fouling Challenge")
            st.plotly_chart(fig, key="fc")

########## Static Period Caluclations #############
        # Step 1: Mark periods of inactivity
        df['inactive'] = df['speed'] < 3
        
        # Step 2: Shift the 'inactive' column to detect changes
        df['inactive_shifted'] = df['inactive'].shift(1, fill_value=False)
        
        # Step 3: Identify start and end of inactive periods
        df['period_start'] = (~df['inactive_shifted'] & df['inactive'])
        df['period_end'] = (df['inactive_shifted'] & ~df['inactive'])
        
        # Step 4: Initialize variables to track inactive periods
        inactive_periods = []
        period_start = None
        
        # Step 5: Loop through the DataFrame to extract periods of inactivity
        for i, row in df.iterrows():
            if row['period_start']:
                period_start = row['DateTime']
            if row['period_end'] and period_start is not None:
                period_end = row['DateTime']
                # Calculate the duration in days
                days = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days
                # Extract the fouling challenge at the end of the period
                fouling_challenge = row['risk']
                # Extract longitude and latitude
                longitude = row['longitude']
                latitude = row['latitude']
                inactive_periods.append((period_start, period_end, days, fouling_challenge, longitude, latitude))
                period_start = None
        
        # Handle case where a period is still ongoing
        if period_start is not None:
            period_end = df['DateTime'].iloc[-1]
            days = (pd.to_datetime(period_end) - pd.to_datetime(period_start)).days
            fouling_challenge = df['risk'].iloc[-1]
            longitude = df['longitude'].iloc[-1]
            latitude = df['latitude'].iloc[-1]
            inactive_periods.append((period_start, period_end, days, fouling_challenge, longitude, latitude))
        
        # Step 6: Create a DataFrame for the inactive periods
        columns = ['Begin', 'End', 'Days', 'Fouling Challenge', 'Longitude', 'Latitude']
        inactive_periods_df = pd.DataFrame(inactive_periods, columns=columns)

########## Static Period Map #############

        inactive_periods_DF2 = inactive_periods_df[inactive_periods_df['Days'] >= 14]
        
        # Define colors for each risk level
        colours = {'null': 'grey', 'VL': 'darkgreen', 'L': '#37ff30', 'M': 'yellow', 'H': 'orange', 'VH': 'red'}
        
        # Create text labels for each point
        text_labels = [
            f"Start: {row['Begin'].date()} Days: {row['Days']}"
            for _, row in inactive_periods_DF2.iterrows()
        ]
        
        # Create the Scattermapbox figure
        fig = go.Figure(go.Scattermapbox(
            lon=inactive_periods_DF2['Longitude'],
            lat=inactive_periods_DF2['Latitude'],
            mode='markers',
            marker={
                'size': 10,
                'color': [colours.get(label, 'grey') for label in inactive_periods_DF2['Fouling Challenge']],
                'opacity': 0.8
            },
            # Add DateTime to hover text
            text=text_labels

        ))
        
        
        # Set map layout
        fig.update_layout(
            mapbox={
                'style': "open-street-map",
                'center': {'lon': inactive_periods_DF2['Longitude'].mean(), 'lat': inactive_periods_DF2['Latitude'].mean()},
                'zoom': 2,
            },
            showlegend=False,
            width=1000,    # Adjust width as needed
            height=1000    # Same height to make it square
        )

        st.subheader("🗺️ Vessel Static Map")
        st.plotly_chart(fig, key="static map")

########## Static Period graph #############

        daycount=np.asarray(pd.cut(inactive_periods_df['Days'].values, [0,13, 21, 30, np.inf], include_lowest=True).value_counts())
        x=['1-13','14-21', '22-30', '>30']
        y=[daycount[0], daycount[1], daycount[2], daycount[3]]
        
        # Create the bar plot with custom colors and outlined bars
        fis = go.Figure(data=[go.Bar(x=x, y=y, 
                                     marker_color=['white', 'grey', 'blue', 'darkblue'],
                                     marker_line_color='black',  # Add black outline
                                     marker_line_width=1)])  # Adjust the width of the outline if needed
        
        
        
        # Add value labels above each bar
        for i, val in enumerate(y):
            fis.add_annotation(
                x=x[i],
                y=val,
                text=f'{int(val)}',
                showarrow=False,
                font=dict(color='black', size=14),
                yshift=10
            )
        
        # Customize plot title and labels
        fis.update_layout(title="STATIONARY PERIODS", xaxis_title='Days At Rest', 
                          template='plotly_white', autosize=False, bargap=0.30,
                          yaxis_title="Frequency",
                          font=dict(color='black',size=14),  # Set the font color to black and size to 12
                          width=500,  # Set the width of the figure to make the graph longer
                          height=500   # Optionally set the height of the figure
        )
        # Display the chart in half the page
        col1, col2 = st.columns([1, 1])  # Two equal-width columns

        with col1:  # or use col2 if you prefer
            st.subheader("STATIONARY PERIODS")
            st.plotly_chart(fis,  key="staticdays")

########## STATIC CHART ########
        
        # Copy and format date columns
        bigstatics2 = inactive_periods_df.copy()
        bigstatics2['Begin'] = bigstatics2['Begin'].dt.strftime('%d %b %Y')
        bigstatics2['End'] = bigstatics2['End'].dt.strftime('%d %b %Y')
        
        # Select relevant columns
        bigstatics2 = bigstatics2[['Days', 'Fouling Challenge', 'Begin', 'End']]
        bigstatics2.sort_values(by='Days', ascending=False, inplace=True)   

        with col2:
            st.subheader("Static Days")
            st.dataframe(bigstatics2)
