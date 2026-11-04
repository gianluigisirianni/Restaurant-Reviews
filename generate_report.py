"""
Google Maps Restaurant Reviews - Generatore automatico di report.

Legge il database SQLite, esegue EDA e analisi testuale e crea:
output/report_restaurant_reviews.html

Uso: python generate_report.py
Il report viene sovrascritto a ogni esecuzione con data e ora aggiornate.
"""

import base64
import io
import re
import sqlite3
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Permette di creare grafici senza aprire finestre
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ------------------------------------------------------------- percorsi
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "dati.db"
OUT = BASE_DIR / "output" / "report_restaurant_reviews.html"

def fig_to_base64(fig):
    """Converte un grafico matplotlib in immagine incorporata nell'HTML."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()

def img_tag(fig, width=760):
    """Crea il tag HTML di un grafico."""
    return (
        f'<img src="data:image/png;base64,{fig_to_base64(fig)}" '
        f'width="{width}" alt="Grafico del report">'
    )

def classifica_sentiment(rating):
    """Stessa regola utilizzata durante l'importazione."""
    if rating >= 4:
        return "positiva"
    if rating == 3:
        return "neutra"
    return "negativa"

def top_bigrammi(testi, n=10):
    """Restituisce le n coppie di parole più frequenti."""
    testo = " ".join(testi.fillna("").astype(str).str.lower())
    parole = re.findall(r"[a-z]+", testo)

    parole_utili = [
        parola for parola in parole
        if parola not in ENGLISH_STOP_WORDS and len(parola) > 2
    ]

    return Counter(zip(parole_utili, parole_utili[1:])).most_common(n)

# ------------------------------------------------------------- 1. dati
print("1/4 - Lettura del database...")

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database non trovato: {DB_PATH}\n"
        "Esegui prima import_to_sqlite.py."
    )

conn = sqlite3.connect(DB_PATH)
try:
    df = pd.read_sql("SELECT * FROM reviews", conn)
finally:
    conn.close()

colonne_richieste = {"business_name", "rating", "rating_category", "text"} 
mancanti = colonne_richieste - set(df.columns)

if mancanti:
    raise ValueError(f"Colonne mancanti nel database: {sorted(mancanti)}")

# Ricalcoliamo il sentiment per rendere il report riproducibile.
df["sentiment"] = df["rating"].apply(classifica_sentiment)

# ------------------------------------------------------------- 2. analisi
print("2/4 - Calcolo statistiche e bigrammi...")

ordine_sentiment = ["positiva", "neutra", "negativa"]
colori = ["#2f9e44", "#868e96", "#e03131"]

conteggi_sentiment = (
    df["sentiment"]
    .value_counts()
    .reindex(ordine_sentiment, fill_value=0)
)

quote_sentiment = (
    conteggi_sentiment / len(df) * 100
).round(1)

per_aspetto = df.groupby("rating_category").agg(
    recensioni=("rating", "size"),
    rating_medio=("rating", "mean"),
    pct_positive=(
        "sentiment",
        lambda valori: valori.eq("positiva").mean() * 100
    ),
    pct_negative=(
        "sentiment",
        lambda valori: valori.eq("negativa").mean() * 100
    ),
).round({
    "rating_medio": 2,
    "pct_positive": 1,
    "pct_negative": 1
})

# L'aspetto più critico viene individuato dai dati, non scritto a mano.
aspetto_critico = per_aspetto["pct_negative"].idxmax()
dati_critici = per_aspetto.loc[aspetto_critico]

recensioni_critiche = df[
    (df["rating_category"] == aspetto_critico)
    & (df["sentiment"] == "negativa")
]

bigrammi_critici = top_bigrammi(recensioni_critiche["text"], n=10)

# ------------------------------------------------------------- 3. grafici
print("3/4 - Creazione dei grafici...")

# Grafico 1: numero di recensioni per sentiment.
fig, ax = plt.subplots(figsize=(7, 4))
conteggi_sentiment.plot(kind="bar", color=colori, ax=ax, rot=0)
ax.set_title("Distribuzione del sentiment")
ax.set_xlabel("")
ax.set_ylabel("Numero di recensioni")
grafico_sentiment = img_tag(fig, width=650)

# Grafico 2: quote di sentiment dentro ogni aspetto.
crosstab = pd.crosstab(
    df["rating_category"],
    df["sentiment"],
    normalize="index"
).reindex(columns=ordine_sentiment, fill_value=0) * 100

fig, ax = plt.subplots(figsize=(9, 4.8))
crosstab.plot(
    kind="bar",
    stacked=True,
    color=colori,
    ax=ax,
    rot=0
)
ax.set_title("Sentiment per aspetto")
ax.set_xlabel("")
ax.set_ylabel("Percentuale di recensioni")
ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
grafico_aspetti = img_tag(fig, width=800)

# ------------------------------------------------------------- 4. HTML
print("4/4 - Generazione del report HTML...")

# Viene calcolata a ogni esecuzione dello script.
data_generazione = datetime.now().strftime("%d/%m/%Y alle %H:%M:%S")

righe_sentiment = "".join(
    f"<tr><td>{sentiment.capitalize()}</td>"
    f"<td>{int(conteggi_sentiment[sentiment])}</td>"
    f"<td>{quote_sentiment[sentiment]:.1f}%</td></tr>"
    for sentiment in ordine_sentiment
)

tabella_aspetti = (
    per_aspetto
    .reset_index()
    .rename(columns={
        "rating_category": "Aspetto",
        "recensioni": "Recensioni",
        "rating_medio": "Rating medio",
        "pct_positive": "% positive",
        "pct_negative": "% negative"
    })
    .to_html(index=False, border=0, classes="dati")
)

if bigrammi_critici:
    righe_bigrammi = "".join(
        f"<tr><td>{' '.join(coppia)}</td><td>{frequenza}</td></tr>"
        for coppia, frequenza in bigrammi_critici
    )

    tabella_bigrammi = f"""
    <table class="dati">
        <tr><th>Bigramma</th><th>Frequenza</th></tr>
        {righe_bigrammi}
    </table>
    """
else:
    tabella_bigrammi = (
        "<p>Non ci sono abbastanza parole per calcolare i bigrammi.</p>"
    )

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Restaurant Reviews — Report automatico</title>
<style>
 body {{
   font-family: 'Segoe UI', sans-serif;
   max-width: 950px;
   margin: 30px auto;
   padding: 0 18px;
   color: #222;
   line-height: 1.55;
 }}
 h1 {{
   border-bottom: 3px solid #d9480f;
   padding-bottom: 8px;
 }}
 h2 {{
   color: #d9480f;
   margin-top: 38px;
 }}
 .meta {{
   color: #555;
 }}
 .box {{
   background: #fff4e6;
   border-left: 4px solid #d9480f;
   padding: 12px 18px;
   margin: 16px 0;
 }}
 table.dati {{
   border-collapse: collapse;
   margin: 14px 0;
 }}
 table.dati th, table.dati td {{
   border: 1px solid #ccc;
   padding: 7px 14px;
   text-align: center;
 }}
 table.dati th {{
   background: #fff4e6;
 }}
 img {{
   display: block;
   max-width: 100%;
   height: auto;
   border: 1px solid #ddd;
   border-radius: 6px;
   margin: 12px 0;
 }}
 code {{
   background: #f1f3f5;
   padding: 2px 5px;
 }}
</style>
</head>
<body>

<h1>Google Maps Restaurant Reviews</h1>
<p class="meta">
Report generato automaticamente il <b>{data_generazione}</b>
da <code>generate_report.py</code>.
Tutti i valori sono calcolati dal database al momento dell'esecuzione.
</p>

<h2>1. Dataset e metodo</h2>
<p>
Il report analizza <b>{len(df):,}</b> recensioni relative a
<b>{df["business_name"].nunique():,}</b> ristoranti
</p>
<p>
Il sentiment è derivato dal rating: <b>positiva</b> per rating ≥ 4,
<b>neutra</b> per rating = 3 e <b>negativa</b> per rating ≤ 2.
Non è quindi un sentiment calcolato automaticamente sul testo.
</p>

<h2>2. Distribuzione del sentiment</h2>
{grafico_sentiment}
<table class="dati">
<tr><th>Sentiment</th><th>Recensioni</th><th>Quota</th></tr>
{righe_sentiment}
</table>

<h2>3. Confronto tra gli aspetti</h2>
{grafico_aspetti}
{tabella_aspetti}

<div class="box">
L'aspetto con la quota più alta di recensioni negative è
<b>{aspetto_critico}</b>: {dati_critici["pct_negative"]:.1f}% negative,
{dati_critici["pct_positive"]:.1f}% positive e rating medio
{dati_critici["rating_medio"]:.2f}.
</div>

<h2>4. Analisi del testo</h2>
<p>
Bigrammi più frequenti nelle <b>{len(recensioni_critiche)}</b>
recensioni negative dell'aspetto <b>{aspetto_critico}</b>:
</p>
{tabella_bigrammi}

<h2>5. Limiti dell'analisi</h2>
<ul>
<li>Il sentiment deriva dal rating, non da un modello linguistico.</li>
<li>I bigrammi mostrano parole ricorrenti, ma non dimostrano una causa.</li>
<li>Le frequenze basse vanno interpretate con cautela.</li>
<li>Per conclusioni operative servono anche il contesto delle recensioni
e un numero sufficiente di osservazioni per ristorante.</li>
</ul>

</body>
</html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8")

print(f"\nReport salvato in: {OUT}")
print(f"Data di generazione: {data_generazione}")

# Apre automaticamente il report nel browser predefinito.
webbrowser.open(OUT.resolve().as_uri())