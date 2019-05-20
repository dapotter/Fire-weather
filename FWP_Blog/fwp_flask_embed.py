from flask import Flask, render_template

import psycopg2

import pandas as pd
import random
import numpy as np
from datetime import datetime

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, DateRangeSlider
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop



# ---------------------------------------------------------------------
# My imports:
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.io import show, output_notebook, curdoc
from bokeh.embed import components
# Color palette stuff:
from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8, cividis
# Basemap stuff:
from bokeh.sampledata.us_states import data as states

from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid)
from bokeh.models.glyphs import VBar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from bokeh.models.ranges import DataRange1d
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Select, Slider
from bokeh.layouts import row, column, widgetbox
# ---------------------------------------------------------------------


from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature

app = Flask(__name__)

# ---------------------------------------------------------------------
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
# ---------------------------------------------------------------------
# def modify_doc(doc):
#     df = sea_surface_temperature.copy()
#     source = ColumnDataSource(data=df)

#     plot = figure(x_axis_type='datetime', y_range=(0, 25), y_axis_label='Temperature (Celsius)',
#                   title="Sea Surface Temperature at 43.18, -70.43")
#     plot.line('time', 'temperature', source=source)

#     def callback(attr, old, new):
#         if new == 0:
#             data = df
#         else:
#             data = df.rolling('{0}D'.format(new)).mean()
#         source.data = ColumnDataSource(data=data).data

#     slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
#     slider.on_change('value', callback)

#     doc.add_root(column(slider, plot))

#     doc.theme = Theme(filename="theme.yaml")
# ---------------------------------------------------------------------
# My fwp_app.py:
def modify_doc(doc):
    conn = connect_to_db()

    # -------------------------------------------
    # QUERY: TIME SERIES OF H500, ERC AT ONE LOCATION
    # Reading data into a list object 'results' directly from postgres fire_weather_db:
    # cur = conn.cursor()
    # sql =  'select id, lat, lon, date, h500, h500_grad_x, erc from narr_erc \
    #         where lat = 39.2549 and lon = 236.314 \
    #         order by id'
    # df = pd.read_sql(sql, conn)
    # cur.close()
    # conn.close()

    # df.set_index('date', inplace=True)
    # # var = 'h500'
    # # var_title = 'H500'
    # # df_var = df[[var]]
    # # var_list = df_var[var].values.tolist()
    # # var_min = 0.99*(min(var_list))
    # # var_max = 1.01*(max(var_list))

    # dict_from_df = {'date': df.index.values, 'y': df['h500'].values}
    # source = ColumnDataSource(dict_from_df)

    #         p.y_range = DataRange1d(start=var_min, end=var_max)
    #         p.yaxis.axis_label = 'Geopotential gradient, gpm/deg'

    # -------------------------------------------
    # # PLOTTING H500 TIME SERIES
    # p = figure(
    #             x_axis_type = 'datetime',
    #             plot_width = 800,
    #             plot_height = 600,
    #             # y_range = [],
    #             x_axis_label = 'Date',
    #             y_axis_label = 'Geopotential height, gpm',
    #             tools = 'pan,zoom_in,zoom_out,save,reset',
    #             title = 'Time Series'
    #             )

    # p.line(
    #         source = source,
    #         x = 'date',
    #         y = 'y',
    #         line_color = 'green',
    #         line_width = 2
    #         )

    # # THIS SLIDER CAUSES THE DATA TO JUMP BACK TO H500 DATA
    # # EVEN WHEN THE SELECT DROPDOWN SHOWS h500_grad_x. THIS
    # # ISSUE IS NOT RESOLVED. IT'S POSSIBLE THAT BOKEH ISN'T
    # # DESIGNED TO HANDLE CHANGING THE COLUMNS TO BE DISPLAYED
    # # IN ONE PLOT.
    # def slider_callback(attr, old, new):
    #     if new == 0:
    #         data = dict_from_df
    #     else:
    #         # df = pd.DataFrame(data=source.data)
    #         df = pd.DataFrame(dict_from_df)
    #         df.set_index('date', inplace=True)

    #         data = df.rolling(new).mean() #'{0}D'.format(new)).mean()

    #     source.data = ColumnDataSource(data=data).data

    # def select_callback(attr, old, new):
    #     if new == 'h500':
    #         # data = df[[new]]
    #         source.data = {'date': df.index.values, 'y': df['h500'].values}
    #         # Rebuilding dict_from_df to pass to slider_callback():
    #         dict_from_df = {'date': df.index.values, 'y': df['h500'].values}


    #     elif new == 'h500_grad_x':
    #         # data = df[[new]]
    #         source.data = {'date': df.index.values, 'y': df['h500_grad_x'].values}
        
    #         # Rebuilding dict_from_df to pass to slider_callback():
    #         dict_from_df = {'date': df.index.values, 'y': df['h500_grad_x'].values}

    #         # Figure p data:
    #         var_list = df['h500_grad_x'].values.tolist()
    #         var_min = 0.99*(min(var_list))
    #         var_max = 1.01*(max(var_list))

    #         p.y_range = DataRange1d(start=var_min, end=var_max)
    #         p.yaxis.axis_label = 'Geopotential gradient, gpm/deg'

    #     dict_from_df = source.data

    # # Widgets
    # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    # slider.on_change('value', slider_callback)
    # select = Select(title="Weather Variable:", value="h500", options=["h500","h500_grad_x"])#["H500", "H500 X Gradient", "H500 Y Gradient", "PMSL", "PMSL X Gradient", "PMSL Y Gradient", "Energy Release Component"])
    # select.on_change('value', select_callback)

    # # Layout
    # # widget_layout = widgetbox(slider, select)
    # # layout = row(slider, p)

    # # Add root:
    # doc.add_root(row(column(slider,select), p))
    # # doc.add_root(row(slider, p))
    # # doc.add_root(row(select, p))

    # # doc.add_root(widget_layout, p)

    # # curdoc().add_root(widget_layout)

    # doc.theme = Theme(filename="fwp_theme.yaml")

    # -------------------------------------------
    # # PLOTTING ERC TIME SERIES
    # p = figure(
    #         x_axis_type = 'datetime',
    #         plot_width = 800,
    #         plot_height = 600,
    #         # y_range = h500_list,
    #         title = 'ERC Time Series',
    #         x_axis_label = 'Date',
    #         y_axis_label = 'ERC, AU',
    #         tools = 'pan,zoom_in,zoom_out,save,reset',
    #         )
    
    # p.line(
    #         source = source,
    #         x = 'date',
    #         y = 'erc',
    #         line_color = 'red',
    #         legend = 'ERC',
    #         line_width=2
    #         )
    # -------------------------------------------
    # SQL QUERY: H500 CONTOUR SINGLE DATE
    # Reading data into a list object 'results' directly from postgres database fire_weather_db:
    cur = conn.cursor()

    # Selecting a specific day:
    # sql =  "select id, lat, lon, date, h500, h500_grad_x, pmsl, pmsl_grad_x, pmsl_grad_y, erc from narr_erc \
    #         where cast(date as date) = '1979-05-15' \
    #         order by id"

    # Selecting all days:
    sql =  "select id, lat, lon, date, h500, h500_grad_x, erc from narr_erc \
            where cast(date as date) >= '1979-01-02' \
            order by id"

    df = pd.read_sql(sql, conn)
    cur.close()
    conn.close()
    # -------------------------------------------
    # # PLOTTING NARR GRID
    # x = df['lon']
    # y = df['lat']

    # p = figure(
    #       plot_width = 800,
    #       plot_height = 600,
    #         title = 'NARR Grid',
    #         x_axis_label = 'Lon',
    #         y_axis_label = 'Lat',
    #         tools = 'pan,zoom_in,zoom_out,save,reset',
    #         )

    # p.circle(x, y, size=2, color="black", alpha=0.5)
    # -------------------------------------------
    # PLOTTING H500 CONTOUR
    var = 'h500'             # e.g. 'h500', 'h500_grad_x', 'erc'
    var_title = 'H500' # e.g. 'H500', 'H500 - X Gradient', 'ERC'

    # Creating mesh of lon and lat then using it to create a mesh of
    # the plotting data (the z data needs to be in meshgrid format,
    # the lat lon data does not):
    lon = df['lon'].drop_duplicates('first').to_numpy()
    lat = df['lat'].drop_duplicates('first').to_numpy()
    lonlon, latlat = np.meshgrid(lon, lat)
    mesh_shape = np.shape(lonlon)

    # If var = 'erc', change -32767 to 0:
    if var == 'erc':
        criteria = df[df['erc'] == -32767].index
        df['erc'].loc[criteria] = 0

    # Getting selected day
    df.set_index('date', inplace=True)
    date_list = df.index.unique().tolist() # df['date'].unique().tolist()
    date_select = date_list[0]
    df = df[[var]]
    df_one_day = df.loc[date_select] #[var]
    d = df_one_day.to_numpy().reshape(mesh_shape)

    # source = ColumnDataSource(df)

    print('Unique date list:\n', date_list)
    print('Selected date:\n', date_select)
    print('df_one_day:\n', df_one_day)
    print('d on a selected day:\n', d)

    # Min and max synoptic variable values:
    # GETS VALUES ONLY FOR ONE DAY OF YEAR:
    var_list = df_one_day[var].values.tolist()
    # GETS VALUES FOR ENTIRE YEAR:
    # var_list = df[var].values.tolist()
    # print('var_list:\n', var_list)
    var_min = min(var_list)
    var_max = max(var_list)
    print('var_min:\n', var_min)

    # Min and max lingitude and latitude for x_range and y_range:
    lon_min = np.min(lon); lon_max = np.max(lon); dw = lon_max - lon_min
    lat_min = np.min(lat); lat_max = np.max(lat); dh = lat_max - lat_min
    print('lat_min:\n', lat_min)

    p = figure(
        #toolbar_location="left",
        title=var_title,
        plot_width=580,
        plot_height=600,
        tooltips=[("lon", "$lon"), ("lat", "$lat"), ("value", "@image")],
        x_range=(lon_min, lon_max),
        y_range=(lat_min, lat_max),
        x_axis_label='Longitude, deg',
        y_axis_label='Latitude, deg'
        )

    # Set palette depending on variable to be plotted
    if var == 'h500_grad_x' or 'h500_grad_y' or 'pmsl_grad_x' or 'pmsl_grad_y':
        # Color maps that make 0 values clear:
        # color_mapper = LinearColorMapper(palette=cividis(256), low=var_min, high=var_max)
        color_mapper = LinearColorMapper(palette="Inferno256", low=var_min, high=var_max)
    else:
        color_mapper = LinearColorMapper(palette="Inferno256", low=var_min, high=var_max)
        # Decent color map: "Spectra11", "Viridis256"

    # Giving a vector of image data for image parameter (contour plot)
    # p.image(image=[d], x=lon_min, y=lat_min, dw=dw, dh=dh, color_mapper=color_mapper)
    source = ColumnDataSource({'image': [d]})
    p.image(image='image', x=lon_min, y=lat_min, dw=dw, dh=dh, color_mapper=color_mapper, source=source)

    # p.x_range.range_padding = p.y_range.range_padding = 0

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0,0)
        )

    p.add_layout(color_bar, 'right')

    # Get state boundaries from state map data imported from Bokeh
    state_lats = [states[code]["lats"] for code in states]
    state_lons = [states[code]["lons"] for code in states]
    # add 360 to adjust lons to NARR grid
    state_lons = np.array([np.array(sublist) for sublist in state_lons])
    state_lons += 360

    # Patch the state and coastal boundaries
    p.patches(state_lons, state_lats, fill_alpha=0.0,
          line_color="black", line_width=2, line_alpha=0.3)

    # select = Select(title="Weather Variable:", value="H500", options=["H500", "H500 X Gradient", "H500 Y Gradient", "PMSL", "PMSL X Gradient", "PMSL Y Gradient", "Energy Release Component"])
    # slider = Slider(start=DateTime(1979,1,2), end=DateTime(1979,12,31), value=DateTime(1979,1,2), step=1, title="Date")
    # slider = Slider(start=1, end=365, step=10, title="Date")

    # def callback(attr, old, new):
    #   points = slider.value
    #   data_points.data = {'x': random(points), 'y': random(points)}

    # To run on bokeh server:
    # bokeh serve --show fwp_app.py

    # # Limit the view to the min and max of the building data
    # p.x_range = DataRange1d(lon_min, lon_max)
    # p.y_range = DataRange1d(lat_min, lat_max)
    # p.xaxis.visible = False
    # p.yaxis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # show(p)

    # output_file("image.html", title="image.py example")

    # show(p)  # open a browser

    # script, div = components(p)

    # The data below is passed to add_user_fwp.html to run when localhost:5000/ is opened.
    # return render_template('add_user_fwp.html', narr_erc_all=narr_erc_all, narr_erc_lat=narr_erc_lat, narr_erc_date=narr_erc_date, script=script, div=div)
    # return render_template('fwp_bokeh_render.html', script=script, div=div, widget_layout=widget_layout)
    # return render_template('bokeh_practice.html', results=results)

    def slider_callback(attr, old, new):
        # if new == 0:
        #     date_select = date_list[0]
        #     print('date_select in callback:\n', date_select)
        #     df_one_day = df.loc[date_select] #[var]
        #     print('df_one_day in callback:\n', df_one_day)
        #     d = df_one_day.to_numpy().reshape(mesh_shape)
        #     print('d in callback:\n', d)

        # else:
        #     date_select = date_list[new]
        #     print('date_select in callback:\n', date_select)
        #     df_one_day = df.loc[date_select] #[var]
        #     print('df_one_day in callback:\n', df_one_day)
        #     d = df_one_day.to_numpy().reshape(mesh_shape)
        #     print('d in callback:\n', d)

            # df = pd.DataFrame(data=source.data)
            # df = pd.DataFrame(dict_from_df)
            # df.set_index('date', inplace=True)
            # data = df.rolling(new).mean() #'{0}D'.format(new)).mean()
        date_select = date_list[slider.value]
        print('date_select in callback:\n', date_select)
        df_one_day = df.loc[date_select] #[var]
        print('df_one_day in callback:\n', df_one_day)
        d = df_one_day.to_numpy().reshape(mesh_shape)
        print('d in callback:\n', d)
        # source.data = ColumnDataSource(data=data).data
        
        source.data = {'image': [d]}

        # ---------------------------------
        # THIS WORKS. UNCOMMENT TO USE:
        # Adjusting color mapper one each slider change:
        var_list = df_one_day[var].values.tolist()
        var_min = min(var_list)
        var_max = max(var_list)
        # color_mapper = LinearColorMapper(palette=cividis(256), low=var_min, high=var_max)
        cm = p.select_one(LinearColorMapper)
        cm.update(low=var_min, high=var_max)
        # ---------------------------------


    slider = Slider(start=0, end=364, value=0, step=1, title="Day of Year")
    # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    slider.on_change('value', slider_callback)

    # def select_callback(attr, old, new):
    #     if new == 'h500':
    #         # data = df[[new]]
    #         source.data = {'date': df.index.values, 'y': df['h500'].values}
    #         # Rebuilding dict_from_df to pass to slider_callback():
    #         dict_from_df = {'date': df.index.values, 'y': df['h500'].values}


    #     elif new == 'h500_grad_x':
    #         # data = df[[new]]
    #         source.data = {'date': df.index.values, 'y': df['h500_grad_x'].values}
        
    #         # Rebuilding dict_from_df to pass to slider_callback():
    #         dict_from_df = {'date': df.index.values, 'y': df['h500_grad_x'].values}

    #         # Figure p data:
    #         var_list = df['h500_grad_x'].values.tolist()
    #         var_min = 0.99*(min(var_list))
    #         var_max = 1.01*(max(var_list))

    #         p.y_range = DataRange1d(start=var_min, end=var_max)
    #         p.yaxis.axis_label = 'Geopotential gradient, gpm/deg'

    #     dict_from_df = source.data

    # # Widgets
    # slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
    # slider.on_change('value', slider_callback)
    # select = Select(title="Weather Variable:", value="h500", options=["h500","h500_grad_x"])#["H500", "H500 X Gradient", "H500 Y Gradient", "PMSL", "PMSL X Gradient", "PMSL Y Gradient", "Energy Release Component"])
    # select.on_change('value', select_callback)

    # # Layout
    # widget_layout = widgetbox(slider, select)
    # layout = row(slider, p)

    # # Add root:
    # doc.add_root(row(column(slider,select), p))
    doc.add_root(row(slider, p))
    # doc.add_root(row(select, p))
    # doc.add_root(widget_layout, p)

    # curdoc().add_root(widget_layout)

    doc.theme = Theme(filename="fwp_theme.yaml")
# ---------------------------------------------------------------------


@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return render_template("fwp_embed.html", script=script, template="Flask")


def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
    server.start()
    server.io_loop.start()

from threading import Thread
Thread(target=bk_worker).start()

if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(port=8000)
