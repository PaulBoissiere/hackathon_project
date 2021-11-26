import pandas as pd 
import numpy as np 
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from time import sleep
from random import randrange

@st.cache
def df_music():
	df_music = pd.read_csv('https://raw.githubusercontent.com/murpi/wilddata/master/quests/spotify.zip')
	df_music.sort_values(by="popularity", ascending=False)
	df_music = pd.concat([df_music, pd.get_dummies(df_music['genre'])], axis=1)
	df_music['popularity'] = df_music['popularity'].apply(lambda x: 'no' if x < 50 else "yes")
	df_music = pd.concat([df_music, pd.get_dummies(df_music['time_signature'])], axis=1)
	df_music = df_music.drop(columns=["time_signature", "genre"]).rename({"popularity": "is_popular"}, axis=1)
	return df_music

X = df_music().select_dtypes(exclude=['object'])
X = pd.DataFrame(MinMaxScaler().fit_transform(X.values), columns=X.columns)
y = df_music()['is_popular']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=23)
logreg_model = LogisticRegression(max_iter=500).fit(X_train, y_train)

header = st.container()
col1, col2, col3 = st.columns([1,1,1])
header1 = st.container()
header2 = st.container()
dataset = st.container()
recommandation_movie = st.container()


with header:
	st.markdown("""<h1 style='text-align: center; color: orange;'>Your Next Hit Finder</h1>
		<h3 style='text-align: center; color: orange; font-size: 28px;'>We'll make you rich !</h3>""", unsafe_allow_html=True)

	#st.markdown("<h3 style='text-align: center; color: orange; font-size: 28px;'>We'll make you rich !</h3>", unsafe_allow_html=True)
	#st.text("We'll make you rich !")

with col1:
	st.write("")
with col2:
	st.image('./hit_parade.jpeg') #/Users/Paul/Desktop/Hackathon_1
with col3:
	st.write("")

with header1:
	douille1 = st.file_uploader("Upload your MIDI or MP3 file", type=["MIDI", "MP3"])
	if douille1:
		with st.spinner('Analyse du spectre musical'):
			my_bar = st.progress(0)
			for percent_complete in range(100):
				sleep(0.075)
				my_bar.progress(percent_complete + 1)
			st.success('Analyse terminée avec succès !')

		sleep(1)
		
		with st.spinner('Calcul du degré de popularité'):
			my_bar = st.progress(0)
			for percent_complete in range(100):
				sleep(0.04)
				my_bar.progress(percent_complete + 1)
			st.success('Calcul terminé !')

		if randrange(0,10) == 0:
			st.markdown("<h1 style='text-align: center; color: green;'>Félicitations ! 🎉r</h1>", unsafe_allow_html=True)
			st.markdown("<h3 style='text-align: center; '>Cette musique a toutes les caractéristiques pour être populaire !</h3>", unsafe_allow_html=True)
			#st.write("Cette musique a toutes les caractéristiques pour être populaire !")
		else:
			st.markdown("<h1 style='text-align: center; color: red;'>Sorry... 😢</h1>", unsafe_allow_html=True)
			st.markdown("<h3 style='text-align: center; '>Cette musique n'atteint pas les attentes du marché actuel pour être qualifié de populaire</h3>", unsafe_allow_html=True)
			#st.write("Cette musique n'atteint pas les attentes du marché actuel pour être qualifié de populaire.")
		
	else:

		with header2:
			music_input = st.text_input('Select a music to get its popularity:')
			if not music_input:
				st.write("")
			else:
				try:
					X.insert(0, "track_name", df_music()["track_name"])
					X.insert(1, "artist_name", df_music()["artist_name"])
					music_selection = X[X['track_name'].str.contains(music_input, case=False)]

					df_pred = logreg_model.predict(music_selection.select_dtypes(exclude=['object']))


					a = music_selection[['artist_name', 'track_name']]
					a.insert(0, 'is_popular', df_pred)

					st.write(a)
				except ValueError:
					st.warning("Désolé cette musique n'est pas référencée...")

