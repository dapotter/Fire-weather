from flask import Flask, render_template, url_for, request
# from flask_bootstrap import Bootstrap
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime # From datetime, import datetime class

import os
import datetime
import time

app = Flask(__name__)
# Bootstrap(app)

# app.config['SQLAlCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# 										# Three slashes are a relative path
# 										# from the current file. site.db will
# 										# be created when we run this python script

# # Creates an SQL alchemy database instance
# # We can represent our database structure as classes (called models)
# # This is an intuitive way to do things
# # Each class will be its own table in the database. The user 
# # class holds our users
# db = SQLAlchemy(app)

# # ------------------------------------------------------------------------------------------------------------------------------------
# class User(db.Model):
# 	id = db.Column(db.Integer, primary_key=True) # Add columns for the table
# 	username = db.Column(db.String(20), unique=True, nullable=False) # Each username must be uniqe, and have to have a username so it can't be null
# 	username = db.Column(db.String(120), unique=True, nullable=False)
# 	image_file = db.Column(db.String(20), nullable=False, default='default.jpg') # The image files will be hashed to 20 characters long so they are all unique
# 	password = db.Column(db.String(60), nullable=False) # unique isn't true because people can have the same password

# 	posts = db.realtionship('Post', backref='author', lazy=True)
# 	# Post attribute has a relationship to the Post model below
# 	# 

# 	# repr for User class
# 	def __repr__(self): # dunder method or magic method. How the object is printed when we print it out
# 		return f"User('{self.username}', '{self.email}', '{self.image_file}')"
# # ------------------------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------------------------
# # Create post class to hold our posts:
# class Post(db.Model):
# 	id = db.Column(db.Integer, primary_key=True)
# 	title = db.Column(db.String(100), nullable=False) # Need to have a title for each of our posts
# 	date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) # If none provided, import current datetime with datetime module
# 																# don't use datetime.utcnow() because this
# 																# would pass in the current time. We want
# 																# to pass in the function, not the time.
# 																# Always use utc times
# 	content = db.Column(db.Text, nullable=False)
# 	# repr for User class
# 	def __repr__(self):
# 		return f"User('{self.title}', '{self.date_posted}')"
# 		# prints out the title and date_posted. Don't print content
# 		# because it could be huge
# # ------------------------------------------------------------------------------------------------------------------------------------

# # ------------------------------------------------------------------------------------------------------------------------------------
# class Author(db.Model) # Users with author posts. Therefore, create a 1-to-many relationship. 1 user can have multiple posts, a post
# 					   # can only have one author

# # LEFT OFF HERE, 4TH INSTALLMENT OF COREY'S FLASK SERIES
# # ------------------------------------------------------------------------------------------------------------------------------------


# List of dicts. Each dict represents a blog post.
# posts = [
# 		{
# 			'author': 'Dan Potter',
# 			'title': 'Fire Weather',
# 			'content': 'First post content',
# 			'date_posted': 'May 6, 2019',
# 		},
# 		{
# 			'author': 'Dan Potter',
# 			'title': 'Campaign 2020',
# 			'content': 'Second post content',
# 			'date_posted': 'May 7, 2019'
# 		}

# ]



@app.route("/")
@app.route("/home")
def home():
	# return "<h1>Home Page</h1>"
	posts = Post.query.all() # Grabs all posts from database
	return render_template('home.html', posts=posts)
										# posts argument and variable are passed in,
										# we will have access to posts variable in
										# our template. We loop through it in template.
@app.route("/about")
def about():
	# return "<h1>About Page</h1>"
	return render_template('about.html', title='About')
										 # title is in the browser tab

@app.route("/campaign")
def campaign():
	post = Post(title='Campaign 2020', content=campaign.html)
	db.session.add(post)
	db.session.commit()
	return render_template('campaign.html', title='Campaign')


if __name__ == '__main__':
	app.run(debug=True)