import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import os

# ---- Helper Functions (mocked) ----
def get_voyage_history(username, password, imo_list, start_date, end_date):
    # Placeholder for actual API call
    return []

def detect_mmsi_changes(voyage_data, end_date):
    # Placeholder for actual MMSI logic
    return [[123456789, datetime.now().isoformat(), datetime.now().isoformat()]]

def fetch_and_combine_ais(username, password, timestamp_changes, start_date, end_date, sixhourly):
    # Placeholder for actual AIS fetching logic
    data = {
        'DateTime': pd.date_range(start=start_date, periods=10, freq='D'),
        'latitude': [10 + i for i in range(10)],
        'longitude': [20 + i for i in range(10)],
        'speed': [0, 0, 3, 4, 0, 0, 6, 7, 0, 1],
    }
    return pd.DataFrame(data)

# ---- Streamlit UI ----
st.title("AIS Risk Processing App")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
imo_input = st.text_input("IMO numbers (comma-separated)")
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=30))
end_date = st.date_input("End Date", datetime.today())
sixhourly = st.checkbox("Fetch every 6 hours")
mmsi_gap_checkbox = st.checkbox("Detect MMSI Gaps")

# Manually loaded correction table (example)
correct_ids = pd.DataFrame({
    'LME_NAME': ['LME A', 'LME B'],
    'LME_NUMBER': [1, 2]
})

if st.button("Fetch Data"):
    with st.spinner("Fetching voyage and AIS data..."):
        try:
            imo_list = list(map(int, imo_input.split(',')))

            if mmsi_gap_checkbox:
                voyage_data = get_voyage_history(username, password, imo_list, start_date, end_date)
                timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())
            else:
                days_ago = end_date - timedelta(days=10)
                voyage_data = get_voyage_history(username, password, imo_list, days_ago, end_date)
                timestamp_changes = detect_mmsi_changes(voyage_data, end_date.isoformat())

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

                required_columns = ['Nov - Jan', 'Feb - Apr', 'May - Jul', 'Aug - Oct']
                for col in required_columns:
                    if col not in LME.columns:
                        raise ValueError(f"Missing required column in LME risk Excel: {col}")

                AIS_long_lat = df_ais[['longitude', 'latitude']]
                AIS_long_lat.columns = ['Longitude', 'Latitude']
                points_cords = [Point(xy) for xy in zip(AIS_long_lat.Longitude, AIS_long_lat.Latitude)]
                Route = gpd.GeoDataFrame(AIS_long_lat, geometry=points_cords, crs='EPSG:4326')

                Route = gpd.sjoin(Route, LEM_gsd_new[['geometry', 'LME_NUMBER']], how="left", predicate='within')
                Route['ID'] = Route['LME_NUMBER']
                Route['Datetime'] = df_ais['DateTime']

                result = pd.merge(Route, LME, how="left", on="ID")
                result['months'] = result['Datetime'].dt.strftime('%b')

                Winter = ['Nov', 'Dec', 'Jan']
                Spring = ['Feb', 'Mar', 'Apr']
                Summer = ['May', 'Jun', 'Jul']
                Autumn = ['Aug', 'Sep', 'Oct']

                def assign_risk(row):
                    if row['months'] in Winter:
                        return row['Nov - Jan']
                    elif row['months'] in Spring:
                        return row['Feb - Apr']
                    elif row['months'] in Summer:
                        return row['May - Jul']
                    else:
                        return row['Aug - Oct']

                df_ais['risk'] = result.apply(assign_risk, axis=1)

                last_known_risk = None
                new_risks = []

                for i in range(len(df_ais)):
                    current_risk = df_ais.at[i, 'risk']
                    current_speed = df_ais.at[i, 'speed']

                    if pd.isna(current_risk):
                        if current_speed == 0:
                            if last_known_risk is not None:
                                new_risks.append(last_known_risk)
                            else:
                                next_known_risk = 'VL'
                                for j in range(i + 1, len(df_ais)):
                                    if not pd.isna(df_ais.at[j, 'risk']):
                                        next_known_risk = df_ais.at[j, 'risk']
                                        break
                                new_risks.append(next_known_risk)
                        else:
                            new_risks.append('VL')
                    else:
                        new_risks.append(current_risk)
                        if current_speed > 0:
                            last_known_risk = current_risk

                df_ais['risk'] = new_risks

                st.dataframe(df_ais)

        except Exception as e:
            st.error(f"An error occurred during data processing: {e}")

