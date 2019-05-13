from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:test123@localhost/flaskmovie' #'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

# ------------------------------------------------------------------------------------------------------------------------------------
class User(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(80), unique=True)
	email = db.Column(db.String(120), unique=True)

	def __init__(self, username, email):
		self.username = username
		self.email = email

	# repr for User class
	def __repr__(self): # dunder method or magic method. How the object is printed when we print it out
		return '<User %r>' % self.username
# ------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
	myUser = User.query.all() # This returns a list of objects. Pass that list of objects to your template using jinga.
	oneItem = User.query.filter_by(username="test2").first() # Pulls just one username from User table. Use .first() in case there are duplicates.
	manyItems = User.query.filter_by(username="test2").all() # Filters by username, grabs every instance of it in the database
	return render_template('add_user.html', myUser=myUser, oneItem=oneItem, manyItems=manyItems) # myUser is passed to template

# Profile page where a user enters a username, than name is then dynamically used to query the database for that user's
# profile information and displays it. A new route is created for this.
@app.route('/profile/<username>') # <> sign is used to pass it an argument, in this case the username. Just a variable name, can be what you want.
def profile(username): # Passing username into the function to be used by the function
	user = User.query.filter_by(username=username).first() # E.g. username=John is then passed into the app.route('/profile/John')
	return render_template('profile.html', user=user) # myUser is passed to template


@app.route('/post_user', methods=['POST']) # For adding a new user to database flaskmovie once the user hits 'submit' on the webpage form
def post_user():
	user = User(request.form['username'], request.form['email'])				# References the User table (class object) which the database schema is centered around
					# Add username and email. The id is added automatically as it's a primary key
					# How do you get the data from the form that's posted?
					# Specify User(request.form['username'], request.form['email'])
					# But once you populate the User object, it's not going to save it
					# to the database, so you need to explicitly add it and then save.
	db.session.add(user)
	db.session.commit()
					# We'll get an error about a view function not returning a valid response.
					# The data wrote to the table 'flaskmovie' but it didn't send anything back,
					# and Flask doesn't like that. Send something back:
	return redirect(url_for('index')) # Redirect to the home page


if __name__ == '__main__':
	app.run(debug=True)