from flask import Flask, render_template

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop

from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature

import psycopg2

import pandas as pd

app = Flask(__name__)

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

print('###################################################')

df = sea_surface_temperature.copy()
source = ColumnDataSource(data=df)
print('df:\n', df)
print('source:\n', source)

data = df.rolling('{0}D'.format(5)).mean()
print('data:\n', data)
source.data = ColumnDataSource(data=data).data
print('source.data:\n', source.data)

# -------------------------------------------
conn = connect_to_db()

# QUERY: TIME SERIES OF H500, ERC AT ONE LOCATION
# Reading data into a list object 'results' directly from postgres fire_weather_db:
cur = conn.cursor()
sql =  'select id, lat, lon, date, h500, erc from narr_erc \
      where lat = 39.2549 and lon = 236.314 \
      order by id'
df = pd.read_sql(sql, conn)
cur.close()
conn.close()

print('###################################################')
df.set_index('date', inplace=True)
df = df[['h500']]
source = ColumnDataSource(df)
print('df:\n', df)
print('source:\n', source)

data = df.rolling(5).mean()
print('data:\n', data)
source.data = ColumnDataSource(data=data).data
print('source.data:\n', source.data)

