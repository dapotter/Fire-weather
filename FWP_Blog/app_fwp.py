from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, func
import pandas as pd

import random
import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.io import show, output_notebook
# Color palette stuff:
from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8

from bokeh.embed import components

from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid, Range1d)
from bokeh.models.glyphs import VBar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:test123@localhost/fire_weather_db' #'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

# ------------------------------------------------------------------------------------------------------------------------------------
class narr_erc(db.Model):
	lat 		= db.Column(db.Float())
	lon 		= db.Column(db.Float())
	date 		= db.Column(db.DateTime())
	h500 		= db.Column(db.Float())
	h500_grad_x = db.Column(db.Float())
	h500_grad_y = db.Column(db.Float())
	pmsl 		= db.Column(db.Float())
	pmsl_grad_x = db.Column(db.Float())
	pmsl_grad_y = db.Column(db.Float())
	cape 		= db.Column(db.Float())
	erc 		= db.Column(db.Integer())
	id   		= db.Column(db.Integer(), primary_key=True)

	# narr_erc class will be initialized with latitude
	# this means that latitude must be requested 
	def __init__(self, lat, lon):#, lon, date, h500, h500_grad_x, h500_grad_y, pmsl, pmsl_grad_x, pmsl_grad_y, cape, erc):
		# self.username = username
		# self.email = email
		self.lat = lat
		self.lon = lon
		# self.date = date
		# self.h500 = h500
		# self.h500_grad_x = h500_grad_x
		# self.h500_grad_y = h500_grad_y
		# self.pmsl = pmsl
		# self.pmsl_grad_x = pmsl_grad_x
		# self.pmsl_grad_y = pmsl_grad_y
		# self.cape = cape
		# self.erc = erc

	# repr for User class
	# repr for lat lon
	def __repr__(self): # dunder method or magic method. How the object is printed when we print it out
		# return '<User %r>' % self.username
		return f"narr_erc('{self.lat}','{self.lon}') #, '{self.h500}', '{self.pmsl}', '{self.cape}', '{self.erc}')"

# ------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
	narr_erc_all = narr_erc.query.limit(5).all() # This returns a list of objects. Pass that list of objects to your template using jinga.
	narr_erc_lat = narr_erc.query.filter_by(lat='39.2549').first()
	narr_erc_date = narr_erc.query.filter(func.date(narr_erc.date) <= '1979-01-02').all()
	return render_template('add_user_fwp.html', narr_erc_all=narr_erc_all, narr_erc_lat=narr_erc_lat, narr_erc_date=narr_erc_date) # myUser is passed to template

@app.route('/pull_data', methods=['POST'])
def pull_data():
	narr_erc_lat_lon = narr_erc(request.form['lat'], request.form['lon']) # References the NARR_ERC_table (class object) which the database schema is centered around
					# Add username and email. The id is added automatically as it's a primary key
					# How do you get the data from the form that's posted?
					# Specify User(request.form['username'], request.form['email'])
					# But once you populate the User object, it's not going to save it
					# to the database, so you need to explicitly add it and then save.
	#db.session.add(narr_erc_lat_lon)
	#db.session.commit()
					# We'll get an error about a view function not returning a valid response.
					# The data wrote to the table 'flaskmovie' but it didn't send anything back,
					# and Flask doesn't like that. Send something back:

	source = ColumnDataSource(narr_erc_lat_lon)

	h500_list = source.data['h500'].tolist()

	p = figure(
			plot_width = 800, 
			plot_height = 600,
			#y_range = car,
			y_range = h500_list,
	        title = 'H500 Time Series',
	        x_axis_label = 'Date',
	        tools = 'pan,box_select,zoom_in,zoom_out,save,reset',
	        )

	# # Square, cirlce and line glyph
	# p1.square(squares_x, squares_y, size = 12, color = 'navy', alpha = 0.6)
	# p1.circle(circles_x, circles_y, size = 12, color = 'red')
	p.line(x='date', y='h500', legend='Test', line_width=2)
	# # output_file('bokeh_practice.html')
	# # show(p)

	# p.hbar(
	# 	#y = car,
	# 	#right = hp,
	#     source = source, # data source
	# 	y = 'Car',
	# 	right = 'Horsepower',
	# 	left = 0,
	# 	height = 0.4,
	# 	color = 'orange',
	# 	fill_alpha = 0.9,
	# 	# Making color bar of shades of blue
	# 	# Title, palette, list
	# 	fill_color=factor_cmap('Car',
	# 							palette=Blues8,
	# 							factors=car_list),
	# 	legend='Car'
	# 	)
	
	# # Add Legend
	# p.legend.orientation='vertical'
	# p.legend.location='top_right'
	# p.legend.label_text_font_size='10px'

	# # Add Tooltips
	# hover = HoverTool()
	# hover.tooltips = """
	# 	<div>
	# 		<h3>@Car</h3>
	# 		<div><strong>Price: </strong>@Price<div>
	# 		<div><strong>HP: </strong>@Horsepower<div>
	# 		<div><img src="@Image" alt="" width="200" /></div>
	# 	</div>
	# """
	# # Above: Any fields you have in the 
	# # csv file can be accessed with @
	# p.add_tools(hover)

	# Print out div and script
	script, div = components(p)
	print(div)
	print(script)

	return render_template("bokeh_practice.html", script=script, div=div, source=source)
	#return redirect(url_for('index')) # Redirect to the home page

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