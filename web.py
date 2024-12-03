import streamlit as st
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import numpy as np
from datetime import datetime as dt
from scipy.interpolate import CubicSpline
from statsmodels.tsa.arima.model import ARIMA

#Import the dataset 
predictions = pd.read_csv('predictions.csv')
crime_monthly = pd.read_csv('crime_monthly.csv')
rent_crime_monthly = pd.read_csv('rent_crime_monthly.csv')

#Unify the different region names
rent_crime_monthly['Location'] = rent_crime_monthly['Location'].replace('Manawatu-Whanganui', 'Manawatu-Wanganui')


crime_monthly['Date'] = pd.to_datetime(crime_monthly['Date'])

#Getting monthly crime
total_sentences_predictions = predictions[predictions['Target'] == 'Total Sentences'].sort_values(['Year','Region'])
total_sentences_predictions = total_sentences_predictions.loc[total_sentences_predictions.groupby(['Year', 'Region'])['MSE'].idxmin()]

total_sentences_predictions['Date'] = pd.to_datetime(total_sentences_predictions['Year'], format='%Y') + pd.offsets.YearBegin(0)
total_sentences_predictions.set_index('Date', inplace=True)
total_sentences_predictions_monthly = pd.DataFrame()

for location in total_sentences_predictions['Region'].unique():
    total_sentences_predictions_location = total_sentences_predictions[total_sentences_predictions['Region'] == location]

    years = total_sentences_predictions_location.index.year
    total_sentences_predictionse_values = total_sentences_predictions_location['Predicted']

    months = pd.date_range(start=str(years.min()), end=str(years.max() + 1), freq='MS')

    spline = CubicSpline(years, total_sentences_predictionse_values)
    interpolated_total_sentences = spline(months.year + months.month / 12)

    interpolated_data = pd.DataFrame({
        'Date': months,
        'Location': location,
        'Crime': interpolated_total_sentences
    })

    total_sentences_predictions_monthly = pd.concat([total_sentences_predictions_monthly, interpolated_data])
    
total_sentences_predictions_monthly.reset_index(drop=True, inplace=True)
total_sentences_predictions_monthly['Location'] = total_sentences_predictions_monthly['Location'].apply(lambda x: x if x == 'ALL' else x.replace(" Region", ""))
total_sentences_predictions_monthly = pd.concat([crime_monthly, total_sentences_predictions_monthly])

#Loaded models downloaded from AWS
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost-model-0')

#Creating location data
location_data = {
    'Location Id': [0, 1, 2, 3, 4, 5 , 7, 8, 9, 13, 14, 15],
    'Location': ['ALL', 'Northland', 'Auckland', 'Waikato', 'Bay of Plenty', 'Gisborne', 
                 'Taranaki', 'Manawatu-Wanganui', 'Wellington', 
                 'Canterbury', 'Otago', 'Southland'],
    'Latitude': [0, -35.3708, -36.8485, -37.7833, -37.6878, -38.6623, 
                 -39.3333, -40.3523, -41.2865, -43.6109, -45.0312, -46.4132],
    'Longitude': [0, 174.5020, 174.7633, 175.2773, 176.1651, 178.0176, 
                  174.0500, 175.6077, 174.7762, 172.6365, 170.3174, 168.3475]
}

location_df = pd.DataFrame(location_data)

#Retain only those locations that are present in several datasets
rent_crime_monthly = rent_crime_monthly[rent_crime_monthly['Location'].isin(location_df['Location'])]
rent_monthly = rent_crime_monthly[['Time Frame','Location','Median Rent']]
rent_monthly['Time Frame'] = pd.to_datetime(rent_monthly['Time Frame'])

st.title("Crime and Rent")

#Getting input
years_range = list(range(2025, 2034))  
months_range = list(range(1, 13))
selected_year = st.selectbox("Year", years_range)
selected_month = st.selectbox("Month", months_range)
selected_location = st.selectbox("Location", location_df['Location'])
location_id = location_df[location_df['Location'] == selected_location]['Location Id'].values[0]

selected_date = pd.to_datetime(f'{selected_year}-{selected_month}-01')

#Disable user input of Crime if "ALL" is selected
if selected_location == "ALL":
    crime = st.number_input("Crime", value=0.0, step=0.1, disabled=True)
else:
    crime = st.number_input("Crime", value=0.0, step=0.1)

if st.button("Predict"):
    # Using ARIMA if "ALL" is selected
    if selected_location == "ALL":
        # Calculate the time difference to the selected date
        last_date = rent_monthly['Time Frame'].max()
        selected_date = pd.to_datetime(f'{selected_year}-{selected_month}-01')
        delta_months = (selected_date.year - last_date.year) * 12 + (selected_date.month - last_date.month)
        
        predictions_per_location = {}
        # Loop through each location to make predictions
        for location in location_df['Location']:
            # Filter data for the specific location
            location_data = rent_monthly[rent_monthly['Location'] == location].set_index('Time Frame')
            location_data = location_data.sort_index()
            loc_df = pd.DataFrame(location_data)
            
            N = 13  # Number of lag values to include
            # Create lag features (previous rent values)
            for i in range(1, N + 1):
                loc_df[f'lag_{i}'] = location_data['Median Rent'].shift(i)

            loc_df.dropna(inplace=True)
            y = loc_df['Median Rent']  # Set target variable 

             # Fit ARIMA model with order (1, 2, 1)
            model = ARIMA(y, order=(1, 2, 1))
            model_fit = model.fit()
            
            # Forecast for the number of months calculated (delta_months)
            forecast = model_fit.get_forecast(steps=delta_months)
            predicted_rent = forecast.predicted_mean # Get predicted rent 
            predictions_per_location[location] = predicted_rent.iloc[-1]  # Store the last predicted rent value
            
        # Getting latitude and longitude    
        predictions_df = pd.DataFrame(list(predictions_per_location.items()), columns=['Location', 'Predicted Rent'])
        predictions_df = predictions_df[predictions_df['Location'] != 'ALL']
        merged_df = pd.merge(predictions_df, location_df, on='Location', how='left')
        
        # Set up the map's view state(New Zealand)
        view_state = pdk.ViewState(latitude=-40.9006, longitude=174.8860, zoom=5)
        # Create a ScatterplotLayer for each location with predicted rent
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=merged_df,
            get_position='[Longitude, Latitude]', # Get the coordinates for each location
            get_color='[200, 30, 0, 160]',
            get_radius=10000,  
            pickable=True
        )
        # Tooltip to display when hovering over a point
        tooltip={
        "html": "<b>City:</b> {Location}<br><b>Average Rent:</b> ${Predicted Rent}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
            }
        }
        
        # Render the map with the scatterplot and tooltip
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ))
    # For a specific location, using model to predict
    else:
        # Filter the dataset for the selected location and date range
        selected_df = total_sentences_predictions_monthly[(total_sentences_predictions_monthly['Date'] <= selected_date) & (total_sentences_predictions_monthly['Location'] == selected_location)]
        
        # Update crime value if provided
        if crime >0 :
            selected_df.loc[selected_df.index[-1], 'Crime'] = crime
            
        # Calculate rolling standard deviations of crime over 3 and 6 months
        selected_df['Crime_Rolling_Std_3'] = selected_df['Crime'].rolling(3,1).std()
        selected_df['Crime_Rolling_Std_6'] = selected_df['Crime'].rolling(6,1).std()
        
        # Get features for the prediction
        features = selected_df.iloc[-1][['Crime_Rolling_Std_3', 'Crime_Rolling_Std_6']]
        features['Location Id'] = location_id
        dtest = xgb.DMatrix([features]) # Create DMatrix for the XGBoost model
         
        # Use the loaded model to predict rent
        predictions = loaded_model.predict(dtest) 
        predicted_rent = predictions[0]
        
         # Get the latitude and longitude for the selected location
        location_latitude = location_df.loc[location_df['Location'] == selected_location, 'Latitude'].values[0]
        location_longitude = location_df.loc[location_df['Location'] == selected_location, 'Longitude'].values[0]
       
        # Set up the map's view state for the specific location
        view_state = pdk.ViewState(latitude=location_latitude, longitude=location_longitude, zoom=10)
        
        # Create a ScatterplotLayer for the selected location
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=[{"Longitude": location_longitude, "Latitude": location_latitude}],
            get_position='[Longitude, Latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=10000,
            pickable=True
        )
        
        # Tooltip to display when hovering over a point
        tooltip={
        "html": f"<b>City:</b> {selected_location}<br><b>Average Rent:</b> ${predicted_rent}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
            }
        }
        
        # Render the map with the scatterplot and tooltip
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ))
    











