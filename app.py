#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.title("NFL Draft Prospects - Predicting Who Will Be a 1st Round Pick")

url = r'https://raw.githubusercontent.com/rickglover41/streamlit_hw4/main/nfl_round_one.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', min_value = 10, max_value = 1000, step = 10)

section = st.sidebar.radio('Choose Application Section', ['Look at the Data', 'Make a Prediction'])

# add the decorator for the cache (cache is an intermediate Python function to learn)
@st.cache
def load_data(num):
	df = pd.read_csv(url, nrows = num)
	return df

@st.cache
def create_grouping(x, y):
	grouping = df.groupby(x)[y].mean()
	return grouping

def load_model():
	with open('pipe1.pkl', 'rb') as pickled_mod:
		model = pickle.load(pickled_mod)
	return model

df = load_data(num_rows)


if section == 'Look at the Data':
	x_axis = st.sidebar.selectbox('Choose column for X-axis', df.select_dtypes(include = np.object).columns.tolist())
	y_axis = st.sidebar.selectbox('Choose column for y-axis', ['Round_One'])
	chart_type = st.sidebar.selectbox('Choose your chart type', ['line', 'bar', 'area'])
	
	if chart_type == 'line':
		grouping = create_grouping(x_axis, y_axis)
		st.line_chart(grouping)
	elif chart_type == 'bar':
		grouping = create_grouping(x_axis, y_axis)
		st.bar_chart(grouping)
	elif chart_type == 'area':
		fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
		st.plotly_chart(fig)
	
	st.write(df)

else:
	st.text('Choose Options to Predict if a Player Would be a 1st Round Pick')
	model = load_model()
	age = st.sidebar.slider('Age', min_value = 18, max_value = 28, value = 22, step = 1)
	forty = st.sidebar.number_input('40 time', min_value = 4, max_value = 6, value = 5, step = 1)
	vertical = st.sidebar.number_input('Vertical Jump', min_value = 70, max_value = 100, value = 80, step = 1)
	bench = st.sidebar.number_input('Bench Press Reps 225', min_value = 0, max_value = 40, value = 20, step = 1)
	bmi = st.sidebar.number_input('BMI', min_value = 20, max_value = 45, value = 30, step = 1)
	three_cone = st.sidebar.slider('3 cone drill', min_value = 5, max_value = 7, value = 5, step = 1)
	school = st.sidebar.selectbox('School Attended', df['School'].unique().tolist())
	position = st.sidebar.selectbox('Position Played', df['Position'].unique().tolist())
	
	sample = {
		'Age': age,
		'School': school,
		'Position': position,
		'Sprint_40yd': forty,
		'Year': None,
		'Height': None,
		'Vertical_Jump': vertical,
		'Bench_Press_Reps': bench,
		'Broad_Jump': None,
		'Agility_3cone': three_cone,
		'Shuttle': None,
		'BMI': bmi,
		'Player_Type': None,
		'Position_Type': None,
		'Weight': None,
		'Player Name': None
	}
	
	sample = pd.DataFrame(sample, index = [0])
	prediction = model.predict(sample)[0]
	
	st.title(f"1st Round Pick: ${int(prediction)}")

	