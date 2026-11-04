import sqlite3
from pathlib import Path
import kagglehub
import pandas as pd

BASE_DIR = Path(__file__).parent

#Download e lettura:
path=kagglehub.dataset_download ( "denizbilginn/google-maps-restaurant-reviews")   
df=pd.read_csv(f"{path}/reviews.csv")

# Colonna sentiment:
def sentiment(r): 
    if r>=4 : return "positiva"
    if r==3: return "neutra"
    return "negativa"

df["sentiment"]=df["rating"].apply(sentiment)

#Srittura nel DB : 
(BASE_DIR/"data").mkdir(exist_ok=True)

conn = sqlite3.connect(BASE_DIR/"data"/"dati.db")  
df.to_sql("reviews", conn, if_exists="replace", index=False)

check= pd.read_sql("SELECT COUNT (*) AS righe FROM reviews", conn) 

conn.close()

print(check)
print('OK-tabella "reviews" in data/dati.db')