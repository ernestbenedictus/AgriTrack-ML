import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

air_df = pd.read_csv("air_df.csv") 

sns.set_theme(style='dark')

st.header('Air Quality Dashboard :sparkles:')

st.subheader('Mean Concentration of Particle Matters by Station')
highest_pm25_station = air_df.loc[air_df['PM2.5'].idxmax(), 'station']
lowest_pm25_station = air_df.loc[air_df['PM2.5'].idxmin(), 'station']
highest_pm10_station = air_df.loc[air_df['PM10'].idxmax(), 'station']
lowest_pm10_station = air_df.loc[air_df['PM10'].idxmin(), 'station']

col3, col4 = st.columns(2)
col3.text(f'Highest PM2.5: {highest_pm25_station}')
col3.text(f'Lowest PM2.5: {lowest_pm25_station}')
col4.text(f'Highest PM10: {highest_pm10_station}')
col4.text(f'Lowest PM10: {lowest_pm10_station}')

fig1, ax1 = plt.subplots(figsize=(20, 10))
station_mean_concentration = air_df.groupby('station')[['PM2.5', 'PM10']].mean().reset_index()
station_mean_concentration['Total'] = station_mean_concentration['PM2.5'] + station_mean_concentration['PM10']
sorted_station_mean = station_mean_concentration.sort_values(by=['Total'])
sns.barplot(x='station', y='Total', data=sorted_station_mean, color='blue', label='Total Particles')
plt.title('Mean Concentration of Particle Matters by Station')
plt.xlabel('Station')
plt.ylabel('Mean Concentration (µg/m³)')
plt.legend()
st.pyplot(fig1)

st.subheader('Average Particle Matters Concentrations per Year')
fig2, ax2 = plt.subplots(figsize=(20, 10))
sns.lineplot(data=air_df.groupby('year')['PM2.5'].mean().reset_index(), x='year', y='PM2.5')
sns.lineplot(data=air_df.groupby('year')['PM10'].mean().reset_index(), x='year', y='PM10')
plt.title('Average Particle Matters Concentrations per Year')
plt.xlabel('Year')
plt.ylabel('Mean Concentration (µg/m³)')
st.pyplot(fig2)

st.subheader('Average Gas Concentrations per Year')
fig3, ax3 = plt.subplots(figsize=(20, 10))
sns.lineplot(data=air_df.groupby('year')['NO2'].mean().reset_index(), x='year', y='NO2', marker='o', label='NO2')
sns.lineplot(data=air_df.groupby('year')['SO2'].mean().reset_index(), x='year', y='SO2', marker='o', label='SO2')
sns.lineplot(data=air_df.groupby('year')['O3'].mean().reset_index(), x='year', y='O3', marker='o', label='O3')
plt.title('Average Gas Concentrations per Year')
plt.xlabel('Year')
plt.ylabel('Mean Concentration (µg/m³)')
col1, col2 = st.columns(2)
col1.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(20, 10))
sns.lineplot(data=air_df.groupby('year')['CO'].mean().reset_index(), x='year', y='CO', marker='o', label='CO')
plt.title('Average CO Concentration per Year')
plt.xlabel('Year')
plt.ylabel('Mean Concentration (µg/m³)')
col2.pyplot(fig4)
