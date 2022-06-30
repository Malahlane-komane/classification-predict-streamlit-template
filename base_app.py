"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

# Streamlit dependencies
from turtle import right
import streamlit as st
import joblib,os
import requests
from streamlit_lottie import st_lottie
from PIL import Image
#import streamlit_theme as stt

# Data dependencies
import pandas as pd
import numpy as np


# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
#clean = pd.read_csv("resources/trainclean.csv")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """


	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Analysis", "About us"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Analysis":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Global warming is the long term rise in the average temperature of the Earth's climate.")
		st.markdown("Human activity could account for 95% of the Earth's climate warming.")
		st.markdown("Our model will assist companies to analyze tweets to establish if a person believes in climate change.")
		

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		st.subheader("EDA Analysis")
		graphs = ["Tweet Sentiment Distribution", "Popular Words for News Tweets","Popular Words for pro Tweets", "Popular Words for Neutral Tweets", "Popular Words for Anti Tweets","word cloud"]
		graphs_choice = st.selectbox(
                "Select a graph", graphs)
		if graphs_choice == "Tweet Sentiment Distribution":
			st.image("resources\imgs\streamlit@.png")
			st.markdown("We have analyzed our data and come to a conclusion that most people believe that climate change is man-made.If you can take a look at the distribution more than 50% of the data is Positively to the fact the climate change is man-made with the next highest bar beginning news which shows that whatever is in the news can affect the beliefs of the other people if we have news saying that climate is man-made we might have a high chance that people will believe that.Anti being the smallest bar it just shows that not a lot of people believe that its man-made.")
		elif graphs_choice == "Popular Words for Anti Tweets":
			st.image("resources\imgs\streamlit5.png")
			st.markdown(" The most popular word in anti is  RT which means retweet,it goes to show that on the anti side people don't believe in man-made climate change they just chose to retweet stuff. Going to the word cloud it shows that Don ald Trump believes that Chinese people are the ones that made climate change for their own benefit.")
		elif graphs_choice == "Popular Words for News Tweets":
			st.image("resources\imgs\streamlit2.png")
			st.markdown("We have analyzed that most words in the news are Trump and Global which might mean that Donald Trump tweets a lot about climate change and global warming then it makes it to the news. ")
		elif graphs_choice == "Popular Words for pro Tweets":
			st.image("resources\imgs\streamlit3.png")
			st.markdown("The most popular words in pro tweets are climate and change which goes to show that most people strongly believe that it's man_made in the analysis Donnald trump is not part of the common words in pro tweets. ")
		elif graphs_choice == "Popular Words for Neutral Tweets":
			st.image("resources\imgs\streamlit4.png")
			st.markdown("The most popular words for neutral tweets are climate, RT and global.")
		elif graphs_choice == "word cloud":
			st.image("resources\imgs\streamlit6.png")
			st.markdown("In word cloud we have an analysis that the most used word throughout the tweets in climate change which shows that people are having conversations about climate change and they are starting to take it serious as it's going to affect the earth in a very harmful way.Which means that people might be trying to find a way to save the earth and reduce greenhouse gases.")
		



	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		
		model_name = ["Linear Support Vector Classifier", "Decision Tree Classifier", "Support Vector Classifier", "Logistic Regression Classifier"]
		#selection = st.sidebar.selectbox("Choose Option", options)
		model_choice = st.selectbox(
                "Select a Classifier  Model", model_name)
	# 	lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_OFBfKg.json")
	# 	left_column, top_right_column = st.columns(2)
	# with top_right_column:
	# 	st_lottie(lottie_coding, height=300, key="coding1")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			if model_choice == "Linear Support Vector Classifier":
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor_lsvc = joblib.load(open(os.path.join("resources/lsvc.pkl"),"rb"))
				prediction = predictor_lsvc.predict(vect_text)
			elif model_choice == "Decision Tree Classifier":
				predictor_tree = joblib.load(open(os.path.join("resources/tree.pkl"),"rb"))
				prediction = predictor_tree.predict(vect_text)
			elif model_choice == "Support Vector Classifier":
				predictor_svm = joblib.load(open(os.path.join("resources/svm.pkl"),"rb"))
				prediction = predictor_svm.predict(vect_text)
			elif model_choice == "Logistic Regression Classifier":
				predictor_lr = joblib.load(open(os.path.join("resources/lr.pkl"),"rb"))
				prediction = predictor_lr.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.

			if prediction ==1 :
				st.success("wow, you believe in climate change :smiley: ")
			elif prediction == 0 :
				st.success("ummh, you don't supports nor refuse the belief of man-made climate change :expressionless: ")
			elif prediction == -1 :
				st.success("Nope, you don't believe in climate change :worried: ")
			elif prediction == 2 :
				st.success("well, your tweet links to factual news about climate change")

	# Building out the "about us" page
	if selection == "About us":
		left_column, right_column = st.columns(2)
		with left_column:
			st.header(" Meet our team")
			st.write("##")
			st.write(
				"""
				komane Malahlane - Software Engineer

				Gabrielle Peria -Data scientist

				Pamela Bokabo - Production Manager

				Tshepiso Puka - Marketing Director

				Dumisani - Sales Manager

				Shalom Mashabane -Data engineer
				""")

		with right_column:
			st.header(" Problem statement")
			st.write("##")
			st.write(
				"""At Greenest Wealth, we want to offer poducts and services that are environmentally friendly. We are set on determining how people perceive climate change and their beliefs on it.
				This can be achieved by creating awareness by decreasing our carbon foot print and promoting sustainable and environmentally friendly products and services, thus increasing insights and expanding our marketing strategies.
				Within the data team, we will build a Machine learning model that is based off of data collected on twitter that has been processed to make predictions of people's sentiments towards climate change.
				""")


		# #creating a contact
		with st.container():
			st.write("---")
			st.header(":mailbox_with_mail: Get In Touch With us!")
			st.write("##")

			contact_form = """"
			<form action="https://formsubmit.co/komanemalahlane@gmail.com" method="POST">
	 			<input type="hidden" name="_captcha" value="false">
     			<input type="text" name="name" placeholder="your name" required>
     			<input type="email" name="email" placeholder="your email" required>
	 			<textarea name="message" placeholder="your message"></textarea>
     			<button type="submit">Send</button>
			</form>
			"""

		st.markdown(contact_form, unsafe_allow_html=True)
		# Use local CSS
		def local_css(file_name):
			with open(file_name) as f:
				st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
		local_css("style/style.css")

	
		with st.container():
			st.header(" :round_pushpin: Find us here")
			st.write("##")
			st.write(
				"""
				Office 102, Thuso house, 100 jorissen str, Braamfontein, 
				Johannesburg, 2001
				""")	

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
	

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_22mjkcbb.json")
left_column, right_column = st.columns(2)
with right_column:
    st_lottie(lottie_coding, height=300, key="coding")

