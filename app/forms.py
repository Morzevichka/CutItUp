from wtforms import StringField, BooleanField, SubmitField, PasswordField
from flask_wtf import FlaskForm
from wtforms.validators import EqualTo, DataRequired, Length

class RegisterForm(FlaskForm):
    username = StringField("Login", validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=4, max=12)])
    password2 = PasswordField("Password", validators=[DataRequired(), EqualTo("password")])
    submit = SubmitField("Sing up")

class LoginForm(FlaskForm):
    username = StringField('Login', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember me')
    submit = SubmitField('Sing in')
