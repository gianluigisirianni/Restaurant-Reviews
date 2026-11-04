# Restaurant Reviews — Aspect & Sentiment Analysis 🍽️

Analisi di recensioni Google Maps di ristoranti (dataset Kaggle): non solo
*quanto* sono negative le recensioni, ma *perché* — dal voto numerico al
contenuto del testo, per produrre raccomandazioni azionabili per un ristoratore.

## Dataset

- Fonte: [Google Maps Restaurant Reviews — Kaggle](https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews)
- File usato: `reviews.csv` — **1.100 recensioni**, **100 ristoranti**, 6 colonne
  (`business_name`, `author_name`, `text`, `photo`, `rating`, `rating_category`)
- Ogni riga è una recensione su un **aspetto specifico** del ristorante:
  `taste` (330), `indoor_atmosphere` (288), `outdoor_atmosphere` (286), `menu` (196)
- Testo sempre presente, lingua inglese

## Metodo

1. **EDA** su rating e aspetti (groupby, describe, crosstab)
2. **Feature engineering**: colonna `sentiment` derivata dal rating
   (4-5 positiva, 3 neutra, 1-2 negativa)
3. **Confronto per aspetti** con crosstab normalizzata (percentuali, non conteggi:
   le categorie hanno dimensioni diverse)
4. **Text mining** sulle recensioni negative del menù: pulizia testo
   (minuscole, regex, stop words inglesi), frequenza parole con
   `collections.Counter`, **bigrammi** per cogliere i temi ("prices high",
   "service fee") invece delle singole parole

## Risultati chiave

- Sentiment complessivo: **776 positive (70%)**, 172 neutre, 152 negative
- **Il menù è l'aspetto più critico**: 16,3% negative (massimo) e 64,3% positive
  (minimo); media rating 3,69 contro 3,94-3,99 degli altri aspetti
- **Perché il menù delude**: i bigrammi dominanti nelle 32 recensioni negative
  sono *"prices high"* e *"service fee"* — il problema è **economico**
  (prezzi percepiti alti, addebiti extra), **non la qualità del cibo**
- Il locale viene lodato anche nelle recensioni negative (*"place nice"*):
  le critiche sono mirate, non disgusto generale

> **Raccomandazione per il ristoratore:** trasparenza su prezzi e commissioni
> di servizio prima di qualsiasi intervento sulla cucina.

## Limiti dichiarati

- **Niente classifiche per ristorante**: con 1-2 recensioni menù per locale,
  una media per-ristorante non è statisticamente affidabile. L'analisi per
  aspetto aggregato (196 recensioni menù) resta solida.
- Il sentiment è derivato dal **rating**, non dal testo: le recensioni negative
  contengono spesso anche lodi (recensioni "miste"), quindi un sentiment
  automatico sul testo sarebbe fuorviante senza modelli dedicati.

## Impatto (stima)

Da 1.100 recensioni grezze a una diagnosi con raccomandazione in ~15 minuti di
pipeline, ripetibile su qualsiasi dataset di recensioni con la stessa struttura:
un'attività di lettura manuale che richiederebbe ore diventa un report
automatico con grafici e temi testuali.

## Come riprodurre

```bash
python import_to_sqlite.py     # reviews.csv -> data/dati.db (tabella "reviews")
streamlit run dashboard.py     # dashboard interattiva (KPI, filtri, bigrammi)
python generate_report.py      # report HTML in output/ con data di generazione
```

Il download del dataset avviene via `kagglehub` (serve account Kaggle configurato).

## Stack

Python · pandas · Streamlit · SQLite · matplotlib · scikit-learn (stop words)
