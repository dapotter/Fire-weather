from flask import Flask, render_template, request, redirect, url_for
# from flask_sqlalchemy import SQLAlchemy
# import sqlalchemy
import psycopg2

import pandas as pd
import random
import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.io import show, output_notebook
from bokeh.embed import components
# Color palette stuff:
from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8
# Basemap stuff:
from bokeh.sampledata.us_states import data as states

from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid)
from bokeh.models.glyphs import VBar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.models.ranges import DataRange1d

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:test123@localhost/fire_weather_db' #'sqlite:////tmp/test.db'

def connect_to_db():
	try:
		conn = psycopg2.connect(
								database='fire_weather_db',
								user='postgres',
							    password='test123',
							    host='127.0.0.1',
							    port='5432')
		return conn
	except:
		print('Can not connect to database')

# db = SQLAlchemy(app)

# ------------------------------------------------------------------------------------------------------------------------------------
# class narr_erc(db.Model):
# 	lat 		= db.Column(db.Float())
# 	lon 		= db.Column(db.Float())
# 	date 		= db.Column(db.DateTime())
# 	h500 		= db.Column(db.Float())
# 	h500_grad_x = db.Column(db.Float())
# 	h500_grad_y = db.Column(db.Float())
# 	pmsl 		= db.Column(db.Float())
# 	pmsl_grad_x = db.Column(db.Float())
# 	pmsl_grad_y = db.Column(db.Float())
# 	cape 		= db.Column(db.Float())
# 	erc 		= db.Column(db.Integer())
# 	id   		= db.Column(db.Integer(), primary_key=True)

# 	# narr_erc class will be initialized with latitude
# 	# this means that latitude must be requested 
# 	def __init__(self, lat, lon):#, lon, date, h500, h500_grad_x, h500_grad_y, pmsl, pmsl_grad_x, pmsl_grad_y, cape, erc):
# 		# self.username = username
# 		# self.email = email
# 		self.lat = lat
# 		self.lon = lon
# 		# self.date = date
# 		# self.h500 = h500
# 		# self.h500_grad_x = h500_grad_x
# 		# self.h500_grad_y = h500_grad_y
# 		# self.pmsl = pmsl
# 		# self.pmsl_grad_x = pmsl_grad_x
# 		# self.pmsl_grad_y = pmsl_grad_y
# 		# self.cape = cape
# 		# self.erc = erc

# 	# repr for User class
# 	# repr for lat lon
# 	def __repr__(self): # dunder method or magic method. How the object is printed when we print it out
# 		# return '<User %r>' % self.username
# 		return f"narr_erc('{self.lat}','{self.lon}'), '{self.h500}', '{self.pmsl}', '{self.cape}', '{self.erc}')"

# ------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
	# narr_erc_all = narr_erc.query.limit(5).all() # This returns a list of objects. Pass that list of objects to your template using jinga.
	# narr_erc_lat = narr_erc.query.filter_by(lat='39.2549').first()
	# narr_erc_date = narr_erc.query.filter(func.date(narr_erc.date) <= '1979-01-02').all()
	# df = pd.read_sql(session.query(narr_erc).filter(func.date(narr_erc.date) <= '1979-01-02').statement, session.bind)
	# df = pd.read_sql(session.query(narr_erc).filter(func.date(narr_erc.date) <= '1979-01-02').statement, session.bind)
	#db.session.add(narr_erc_lat_lon)
	#db.session.commit()
	

	# fetchall() is one way to get data from a cursor after a query
	# results = cur.fetchall()

	conn = connect_to_db()

	# -------------------------------------------
	# # QUERY: TIME SERIES OF H500, ERC AT ONE LOCATION
	# # Reading data into a list object 'results' directly from postgres fire_weather_db:
	# cur = conn.cursor()
	# sql =  'select id, lat, lon, date, h500, erc from narr_erc \
	# 		where lat = 39.2549 and lon = 236.314 \
	# 		order by id'
	# df = pd.read_sql(sql, conn)
	# cur.close()
	# conn.close()
	# source = ColumnDataSource(df)
	# -------------------------------------------
	# # PLOTTING H500 TIME SERIES
	# p = figure(
	# 		x_axis_type = 'datetime',
	# 		plot_width = 800,
	# 		plot_height = 600,
	# 		# y_range = h500_list,
	#         title = 'H500 Time Series',
	#         x_axis_label = 'Date',
	#         y_axis_label = 'Geopotential height, gpm',
	#         tools = 'pan,zoom_in,zoom_out,save,reset',
	#         )
	#
	# p.line(
	#     source = source,
	#     x = 'date',
	# 	y = 'h500',
	# 	line_color = 'green',
	# 	legend = 'H500',
	# 	line_width = 2
	# 	)
	# -------------------------------------------
	# # PLOTTING ERC TIME SERIES
	# p = figure(
	# 		x_axis_type = 'datetime',
	# 		plot_width = 800,
	# 		plot_height = 600,
	# 		# y_range = h500_list,
	#         title = 'ERC Time Series',
	#         x_axis_label = 'Date',
	#         y_axis_label = 'ERC, AU',
	#         tools = 'pan,zoom_in,zoom_out,save,reset',
	#         )
	#
	# p.line(
	#     source = source,
	#     x = 'date',
	# 	y = 'erc',
	# 	line_color = 'red',
	# 	legend = 'ERC',
	# 	line_width=2
	# 	)
	# -------------------------------------------
	# SQL QUERY: H500 CONTOUR SINGLE DATE
	# Reading data into a list object 'results' directly from postgres fire_weather_db:
	cur = conn.cursor()
	sql =  "select id, lat, lon, date, h500, erc from narr_erc \
			where cast(date as date) = '1979-05-15' \
			order by id"
	df = pd.read_sql(sql, conn)
	cur.close()
	conn.close()
	source = ColumnDataSource(df)
	# -------------------------------------------
	# # PLOTTING NARR GRID
	# x = df['lon']
	# y = df['lat']
	
	# p = figure(
	# 		plot_width = 800,
	# 		plot_height = 600,
	#         title = 'NARR Grid',
	#         x_axis_label = 'Lon',
	#         y_axis_label = 'Lat',
	#         tools = 'pan,zoom_in,zoom_out,save,reset',
	#         )

	# p.circle(x, y, size=2, color="black", alpha=0.5)
	# -------------------------------------------
	# PLOTTING H500 CONTOUR

	# determine range to print based on min, max lat and long of the data
	# margin = .2 # buffer to add to the range
	# lat_min = min(lat) - margin
	# lat_max = max(lat) + margin
	# long_min = min(long) - margin
	# long_max = max(long) + margin

	lon = df['lon'].drop_duplicates('first').to_numpy()
	lat = df['lat'].drop_duplicates('first').to_numpy()
	lonlon, latlat = np.meshgrid(lon, lat)
	mesh_shape = np.shape(lonlon)
	d = df['h500'].to_numpy().reshape(mesh_shape)

	lon_min = np.min(lon); lon_max = np.max(lon); dw = lon_max - lon_min
	lat_min = np.min(lat); lat_max = np.max(lat); dh = lat_max - lat_min

	p = figure(
		#toolbar_location="left",
		plot_width=700,
	    plot_height=700,
		tooltips=[("lon", "$lon"), ("lat", "$lat"), ("value", "@image")],
		x_range=(lon_min, lon_max),
		y_range=(lat_min, lat_max)
		)

	# Giving a vector of image data for image parameter (contour plot)
	p.image(image=[d], x=lon_min, y=lat_min, dw=dw, dh=dh, palette="Spectral11")
	# p.x_range.range_padding = p.y_range.range_padding = 0

	# get state boundaries from state map data imported from Bokeh
	state_lats = [states[code]["lats"] for code in states]
	state_lons = [states[code]["lons"] for code in states]
	# add 360 to adjust lons to NARR grid
	state_lons = np.array([np.array(sublist) for sublist in state_lons])
	state_lons += 360

	p.patches(state_lons, state_lats, fill_alpha=0.0,
	      line_color="black", line_width=2, line_alpha=0.3)

	# # Limit the view to the min and max of the building data
	# p.x_range = DataRange1d(lon_min, lon_max)
	# p.y_range = DataRange1d(lat_min, lat_max)
	# p.xaxis.visible = False
	# p.yaxis.visible = False
	# p.xgrid.grid_line_color = None
	# p.ygrid.grid_line_color = None

	# show(p)



	# output_file("image.html", title="image.py example")

	# show(p)  # open a browser

	script, div = components(p)


	# The data below is passed to add_user_fwp.html to run when localhost:5000/ is opened.
	# return render_template('add_user_fwp.html', narr_erc_all=narr_erc_all, narr_erc_lat=narr_erc_lat, narr_erc_date=narr_erc_date, script=script, div=div)
	return render_template('fwp_bokeh_render.html', script=script, div=div, df=df, state_lats=state_lats, state_lons=state_lons)
	# return render_template('bokeh_practice.html', results=results)





# @app.route('/pull_data', methods=['POST'])
# def pull_data():

# 	# narr_erc_lat_lon = narr_erc(request.form['lat'], request.form['lon']) # References the NARR_ERC_table (class object) which the database schema is centered around
# 					# Add username and email. The id is added automatically as it's a primary key
# 					# How do you get the data from the form that's posted?
# 					# Specify User(request.form['username'], request.form['email'])
# 					# But once you populate the User object, it's not going to save it
# 					# to the database, so you need to explicitly add it and then save.
# 	#db.session.add(narr_erc_lat_lon)
# 	#db.session.commit()
# 					# We'll get an error about a view function not returning a valid response.
# 					# The data wrote to the table 'flaskmovie' but it didn't send anything back,
# 					# and Flask doesn't like that. Send something back:

# 	conn = psycopg2.connect("dbname=fire_weather_db user=postgres")

# 	# get h500
# 	narr_erc_h500 = pd.read_sql("""
# 		SELECT date, lat, lon, h500 FROM narr_erc
# 		WHERE lat=39.2549 AND lon=236.614;
# 		""", conn)
# 	narr_erc_h500.set_index('date', inplace=True)

# 	# source = ColumnDataSource(narr_erc_lat_lon)
# 	source = ColumnDataSource(narr_erc_h500)

# 	h500_list = source.data['h500'].tolist()

# 	p = figure(
# 			plot_width = 800, 
# 			plot_height = 600,
# 			#y_range = car,
# 			y_range = h500_list,
# 	        title = 'H500 Time Series',
# 	        x_axis_label = 'Date',
# 	        tools = 'pan,box_select,zoom_in,zoom_out,save,reset',
# 	        )

# 	# # Square, cirlce and line glyph
# 	# p1.square(squares_x, squares_y, size = 12, color = 'navy', alpha = 0.6)
# 	# p1.circle(circles_x, circles_y, size = 12, color = 'red')
# 	p.line(x='date', y='h500', legend='Test', line_width=2)
# 	# # output_file('bokeh_practice.html')
# 	# # show(p)

# 	# p.hbar(
# 	# 	#y = car,
# 	# 	#right = hp,
# 	#     source = source, # data source
# 	# 	y = 'Car',
# 	# 	right = 'Horsepower',
# 	# 	left = 0,
# 	# 	height = 0.4,
# 	# 	color = 'orange',
# 	# 	fill_alpha = 0.9,
# 	# 	# Making color bar of shades of blue
# 	# 	# Title, palette, list
# 	# 	fill_color=factor_cmap('Car',
# 	# 							palette=Blues8,
# 	# 							factors=car_list),
# 	# 	legend='Car'
# 	# 	)
	
# 	# # Add Legend
# 	# p.legend.orientation='vertical'
# 	# p.legend.location='top_right'
# 	# p.legend.label_text_font_size='10px'

# 	# # Add Tooltips
# 	# hover = HoverTool()
# 	# hover.tooltips = """
# 	# 	<div>
# 	# 		<h3>@Car</h3>
# 	# 		<div><strong>Price: </strong>@Price<div>
# 	# 		<div><strong>HP: </strong>@Horsepower<div>
# 	# 		<div><img src="@Image" alt="" width="200" /></div>
# 	# 	</div>
# 	# """
# 	# # Above: Any fields you have in the 
# 	# # csv file can be accessed with @
# 	# p.add_tools(hover)

# 	# Print out div and script
# 	script, div = components(p)
# 	print(div)
# 	print(script)

# 	return render_template("bokeh_practice.html", script=script, div=div, source=source)
# 	#return redirect(url_for('index')) # Redirect to the home page

# def create_figure():
# 	# Create a blank figure with labels

# 	# Data
# 	# x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# 	# y = [4,6,2,8,5,10,9,1,5,8,1,2,7,7,4]
# 	# squares_x = [1, 3, 4, 5, 8]
# 	# squares_y = [8, 7, 3, 1, 10]
# 	# circles_x = [9, 12, 4, 3, 15]
# 	# circles_y = [8, 4, 11, 6, 10]

# 	df = pd.read_csv('cars.csv')
# 	# car = df['Car']
# 	# hp = df['Horsepower']
# 	# Instead of specifying individual columns 'car' and 'hp'
# 	# as above, create a ColumnDataSource
# 	# from the dataframe:
# 	source = ColumnDataSource(df)
# 	# Pass 'source' into the hbar glyph below
# 	# Then get rid of column names y=car and replace
# 	# with y = 'Car' and right = 'Horsepower'

# 	# p1 = figure(
# 	# 		plot_width = 600, 
# 	# 		plot_height = 600, 
# 	#         title = 'Bokeh Practice',
# 	#         x_axis_label = 'X axis',
# 	#         y_axis_label = 'Y axis'
# 	#         )
# 	# p2 = figure(
# 	# 		plot_width = 600, 
# 	# 		plot_height = 600, 
# 	#         title = 'Bokeh Practice',
# 	#         x_axis_label = 'X axis',
# 	#         y_axis_label = 'Y axis'
# 	#         )
	
# 	car_list = source.data['Car'].tolist()

# 	p = figure(
# 			plot_width = 800, 
# 			plot_height = 600,
# 			#y_range = car,
# 			y_range = car_list,
# 	        title = 'Cars',
# 	        x_axis_label = 'HP',
# 	        tools = 'pan,box_select,zoom_in,zoom_out,save,reset',
# 	        )

# 	# # Square, cirlce and line glyph
# 	# p1.square(squares_x, squares_y, size = 12, color = 'navy', alpha = 0.6)
# 	# p1.circle(circles_x, circles_y, size = 12, color = 'red')
# 	# p2.line(x,y,legend='Test', line_width=2)
# 	# # output_file('bokeh_practice.html')
# 	# # show(p)

# 	p.hbar(
# 		#y = car,
# 		#right = hp,
# 	    source = source, # data source
# 		y = 'Car',
# 		right = 'Horsepower',
# 		left = 0,
# 		height = 0.4,
# 		color = 'orange',
# 		fill_alpha = 0.9,
# 		# Making color bar of shades of blue
# 		# Title, palette, list
# 		fill_color=factor_cmap('Car',
# 								palette=Blues8,
# 								factors=car_list),
# 		legend='Car'
# 		)
	
# 	# # Add Legend
# 	# p.legend.orientation='vertical'
# 	# p.legend.location='top_right'
# 	# p.legend.label_text_font_size='10px'

# 	# # Add Tooltips
# 	# hover = HoverTool()
# 	# hover.tooltips = """
# 	# 	<div>
# 	# 		<h3>@Car</h3>
# 	# 		<div><strong>Price: </strong>@Price<div>
# 	# 		<div><strong>HP: </strong>@Horsepower<div>
# 	# 		<div><img src="@Image" alt="" width="200" /></div>
# 	# 	</div>
# 	# """
# 	# # Above: Any fields you have in the 
# 	# # csv file can be accessed with @
# 	# p.add_tools(hover)

# 	# Print out div and script
# 	script, div = components(p)
# 	print(div)
# 	print(script)

# 	return source, p


if __name__ == '__main__':
	app.run(debug=True)