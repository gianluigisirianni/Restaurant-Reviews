import sqlite3
import pandas as pd
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

st.set_page_config(page_title="Restaurant Reviews", page_icon = "🍽️", layout= "wide")

conn = sqlite3.connect("data/dati.db")

df= pd.read_sql("SELECT * FROM reviews", conn)

conn.close()

st.title("Restaurant Reviews — Dashboard")

aspetti = sorted(df["rating_category"].unique())

scelta_aspetto = st.sidebar.selectbox("Aspetto:",aspetti)

df_aspetto= df[df["rating_category"]==scelta_aspetto]

sentiments= ["positiva", "neutra", "negativa"]
scelta_sent= st.sidebar.selectbox("Sentiment:", sentiments)

df_filtrato= df_aspetto[df_aspetto["sentiment"]==scelta_sent]

col1, col2, col3= st.columns(3)

col1.metric("Recensioni", len(df_filtrato))
col2.metric("Rating medio", round(df_filtrato["rating"].mean(),2))
col3.metric("% positive sull'aspetto", round((df_aspetto["sentiment"]=="positiva").mean()*100,1))


st.subheader("Sentiment per aspetto (quote)")

ct=pd.crosstab(df["rating_category"],df["sentiment"], normalize="index") 

fig= ct.plot (kind="bar", stacked=True, rot=0).get_figure()
st.pyplot(fig)

st.subheader(f"Top bigrammi - {scelta_aspetto}, {scelta_sent}")

testo = " ".join(df_filtrato["text"].str.lower())
parole= re.findall(r"[a-z]+", testo)
utili = [p for p in parole if p not in ENGLISH_STOP_WORDS and len(p)>2]

bigrammi = Counter(zip(utili, utili[1:]))

for coppia, n in bigrammi.most_common(10):
    st.write(n," ".join(coppia))

st.caption(f"Filtro attivo: {len(df_filtrato)} recensioni su {len(df_aspetto)} dell'aspetto")