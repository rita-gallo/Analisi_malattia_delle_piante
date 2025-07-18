
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import warnings
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")

# Caricamento del dataset
df = pd.read_csv("plant_disease_dataset.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.describe())

df["disease_present"].value_counts()

# Configurazione per i grafici
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# Inizializza l'oggetto caricando i dati e richiamando il metodo
class plant_analyzer:
    def __init__(self, file_path):
        """Inizializza l'analizzatore caricando i dati"""
        self.df = pd.read_csv(file_path)
        self.prepare_data()
# Rendi disponibili i dati globalmente 
        global df_global, analyzer_global
        df_global = self.df
        analyzer_global = self

# Rimuove valori mancanti e codifica la varialible target
    def prepare_data(self):
        """Prepara e pulisce i dati"""
        print(" Preparazione dei dati...")

# Rimuove i valori mancanti
        self.df = self.df.dropna()

 # Codifica la variabile target
        label_encoder = LabelEncoder()
        self.df["disease_present_encoded"] = label_encoder.fit_transform(self.df["disease_present"])

        print(" Dati pronti per l'analisi!")

# Analisi esplorativa dei dati con statistiche descrittive, distribuzione della variabile target, correlazioni tra variabili, chiama la funzione per i grafici
    def exploratory_analysis(self):
        """Analisi esplorativa dei dati"""
        print("\n" + "=" * 50)
        print(" ANALISI ESPLORATIVA DEI DATI AGRONOMICI")
        print("=" * 50)

        stats_cols = ["temperature", "humidity", "rainfall", "soil_pH"]

        print("\n Statistiche descrittive:")
        print(self.df[stats_cols].describe().round(2))

        print("\n Distribuzione della presenza di malattia:")
        print(self.df["disease_present"].value_counts())

        print("\n Matrice di correlazione:")
        corr_matrix = self.df[stats_cols + ["disease_present_encoded"]].corr()
        print(corr_matrix.round(2))

        self.create_visualizations()

# Carea 8 grafici: istogrammi per temperatura,humidity,rainfall e soil_pH, boxplot per disease_present, heatmap delle correlazioni. visualizzare in modo chiaro come si relazionano le variabili con la malattia
    def create_visualizations(self):
        """Crea le visualizzazioni dei dati"""
        print("\n Creazione visualizzazioni...")

        fig, axs = plt.subplots(4, 2, figsize=(18, 20))

 # Temperatura
        sns.histplot(self.df["temperature"], bins=30, color="tomato", kde=True, ax=axs[0, 0])
        axs[0, 0].set_title("Distribuzione della Temperatura (°C)")
        axs[0, 0].set_xlabel("Temperatura (°C)")
        axs[0, 0].set_ylabel("Frequenza")

# Umidità
        sns.histplot(self.df["humidity"], bins=30, color="skyblue", kde=True, ax=axs[0, 1])
        axs[0, 1].set_title("Distribuzione dell'Umidità (%)")
        axs[0, 1].set_xlabel("Umidità (%)")
        axs[0, 1].set_ylabel("Frequenza")

 #  Piovosità
        sns.histplot(self.df["rainfall"], bins=30, color="green", kde=True, ax=axs[1, 0])
        axs[1, 0].set_title("Distribuzione della Piovosità (mm)")
        axs[1, 0].set_xlabel("Piovosità (mm)")
        axs[1, 0].set_ylabel("Frequenza")

 #  pH Suolo
        sns.histplot(self.df["soil_pH"], bins=30, color="purple", kde=True, ax=axs[1, 1])
        axs[1, 1].set_title("Distribuzione del pH del Suolo")
        axs[1, 1].set_xlabel("pH Suolo")
        axs[1, 1].set_ylabel("Frequenza")

 #  Boxplot Temp vs Malattia
        sns.boxplot(x="disease_present", y="temperature", data=self.df, ax=axs[2, 0])
        axs[2, 0].set_title("Temperatura vs Presenza Malattia")
        axs[2, 0].set_xlabel("Malattia Presente")
        axs[2, 0].set_ylabel("Temperatura (°C)")

 # Boxplot Umidità vs Malattia
        sns.boxplot(x="disease_present", y="humidity", data=self.df, ax=axs[2, 1])
        axs[2, 1].set_title("Umidità vs Presenza Malattia")
        axs[2, 1].set_xlabel("Malattia Presente")
        axs[2, 1].set_ylabel("Umidità (%)")

 #  Heatmap correlazioni
        sns.heatmap(
            self.df[["temperature", "humidity", "rainfall", "soil_pH", "disease_present_encoded"]].corr(),
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            ax=axs[3, 0]
        )
        axs[3, 0].set_title("Matrice di Correlazione")

 #  Slot vuoto per layout
        axs[3, 1].axis("off")
        axs[3, 1].text(0.5, 0.5, "Analisi Agronomica", fontsize=14, ha="center", va="center")

        plt.tight_layout(pad=3.0)
        plt.savefig("crop_disease_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(" Visualizzazioni create e salvate come 'crop_disease_analysis.png'")

plt.figure(figsize=(10,6))
sns.countplot(data=df,x="disease_present",palette="Set1")
plt.title("Distribution of Disease Presence")
plt.show()

def mean_by_disease_status(column: str) -> str:
    if column not in df_global.columns:
        return f"Colonna '{column}' non trovata nel dataset."
    if not pd.api.types.is_numeric_dtype(df_global[column]):
        return f"La colonna '{column}' non è numerica."
    
    means = df_global.groupby("malattia")[column].mean()
    
    result = []
    for status, mean_val in means.items():
        result.append(f"La media di '{column}' per le piante {status} è {mean_val:.2f}")
    return "\n".join(result)


#visualizzazioni valori massimi
df.groupby('disease_present')[['temperature', 'humidity', 'rainfall', 'soil_pH']].max().T.plot(kind='bar')
plt.title("Valori massimi per gruppo di malattia")
plt.ylabel("Valore massimo")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Preparazione dei dati MachineLearning selezionando variabili indipendenti(X) e target codificato(y) mostra l'equilibrio delle classi nel target
def prepare_ml_features(self):
        """Prepara le features per il machine learning"""
        print("\n Preparazione features per Machine Learning...")

        feature_columns = ["temperature", "humidity", "rainfall", "soil_pH"]
        X = self.df[feature_columns]
        y = self.df["disease_present_encoded"]

        print(f" Dataset preparato: {len(X)} campioni, {len(feature_columns)} features")
        print(f" Target bilanciamento:")
        print(y.value_counts(normalize=True).round(3))

        return X, y, feature_columns
    
    # Addestramento del modello di classificazione LogisticRegression per classificare la presenza di malattia. calcola:accuratezza, confusion matrix, classification report, importanza delle features
def train_prediction_model(self):
        """Addestra il modello di classificazione per presenza malattia"""
        print("\n MODELLO DI PREDIZIONE PRESENZA DI MALATTIA")
        print("=" * 50)

        X, y, feature_columns = self.prepare_ml_features()

        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

# Usa regressione logistica per classificazione binaria
        print(" Addestramento del modello (Logistic Regression)...")
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

       
        y_pred = model.predict(X_test)

# Metriche di performance
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        print(f"\n PERFORMANCE DEL MODELLO:")
        print(f" Accuratezza: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)

# Coefficienti di feature importance
        feature_importance = pd.DataFrame(
            {"Feature": feature_columns, "Coefficient": model.coef_[0]}
        ).sort_values(by="Coefficient", key=abs, ascending=False)

        print("\n Importanza delle Features (Coefficiente della regressione):")
        for idx, row in feature_importance.iterrows():
            print(f"{row['Feature']:<15}: {row['Coefficient']:.4f}")

        self.plot_classification_results(X_test, y_test, y_pred, feature_importance)
        return model
              

def create_visualizations(self):
    """Crea le visualizzazioni dei dati separatamente"""
    print("\nCreazione visualizzazioni...")

#distribuzioni
plt.figure(figsize=(12, 10))
plt.suptitle("Distribuzioni delle Variabili", fontsize=16)

plt.subplot(2, 2, 1)
sns.histplot(df["temperature"], bins=30, kde=True, color="salmon")
plt.title("Temperatura (°C)")

plt.subplot(2, 2, 2)
sns.histplot(df["humidity"], bins=30, kde=True, color="skyblue")
plt.title("Umidità (%)")

plt.subplot(2, 2, 3)
sns.histplot(df["rainfall"], bins=30, kde=True, color="green")
plt.title("Piovosità (mm)")

plt.subplot(2, 2, 4)
sns.histplot(df["soil_pH"], bins=30, kde=True, color="purple")
plt.title("pH del Suolo")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#boxplot vs malattia
plt.figure(figsize=(10, 5))
plt.suptitle("Boxplot vs Malattia", fontsize=16)

plt.subplot(1, 2, 1)
sns.boxplot(x="disease_present", y="temperature", data=df)
plt.title("Temperatura vs Malattia")

plt.subplot(1, 2, 2)
sns.boxplot(x="disease_present", y="humidity", data=df)
plt.title("Umidità vs Malattia")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#matrice di correlazione
plt.figure(figsize=(6, 5))
plt.title("Matrice di Correlazione", fontsize=14)

corr = df[["temperature", "humidity", "rainfall", "soil_pH", "disease_present"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, fmt=".2f")

plt.tight_layout()
plt.show()

print("Visualizzazioni completate.")

# if __name__ == "__main__":
#     analyzer = plant_analyzer("plant_disease_dataset.csv")
#     analyzer.exploratory_analysis()


#  Regressione Logistica

print("\n Regressione Logistica:")
features = ['temperature', 'humidity', 'rainfall', 'soil_pH']
target = 'disease_present'

X_multi = df[features]
y_multi = df[target]

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#  Valutazione del Modello

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.tight_layout()
plt.show()


#  Coefficienti del modello

coeff_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
})
coeff_df['Importance'] = coeff_df['Coefficient'].abs()
coeff_df['Influenza'] = coeff_df['Coefficient'].apply(lambda x: 'Positiva' if x > 0 else 'Negativa')
coeff_df = coeff_df.sort_values(by='Importance', ascending=True)

print("\n Coefficienti del modello (ordinati per importanza):")
print(coeff_df)

#  grafico dei coefficienti

plt.figure(figsize=(8, 6))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df, hue='Influenza', palette={'Positiva': 'green', 'Negativa': 'red'})
plt.axvline(0, color='grey', linewidth=1)
plt.title(" Coefficienti della Regressione Logistica")
plt.xlabel("Valore del Coefficiente")
plt.ylabel("Variabile")
plt.legend(title="Influenza")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#aggiungere colonna 
# Lista degli indicatori
indicatori = ['humidity', 'rainfall', 'soil_pH', 'temperature']

# Tipi associati agli indicatori
tipi = ['positivo', 'positivo', 'negativo', 'non_rilevante']

# Creo un DataFrame con queste informazioni
df_indicatori = pd.DataFrame({
    'indicatore': indicatori,
    'tipo_indicatore': tipi
})

print(df_indicatori)



# RISCHIO FUTURO DELLE MALATTIA DELLE PIANTE RISPETTO IL CAMBIAMENTO CLIMATICO

# Feature engineering
df_fe = df.copy()
# nuova feature moltiplicando temperatura e umidità: può evidenziare condizioni climatiche estreme.
df_fe["temp_humidity_interaction"] = df_fe["temperature"] * df_fe["humidity"]   
# indicatore di “pioggia relativa alla temperatura”, utile per valutare condizioni anomale                                
df_fe["rainfall_per_temp"] = df_fe["rainfall"] / (df_fe["temperature"] + 1)
# Flag binario (0 o 1) che indica se la pioggia supera una soglia alta
df_fe["high_rain_flag"] = (df_fe["rainfall"] > 10).astype(int)
# Flag binario per identificare alta umidità
df_fe["high_humidity_flag"] = (df_fe["humidity"] > 80).astype(int)
# Flag binario per pH acido del suolo (basso, sotto 5.5), condizione sfavorevole per molte piante
df_fe["low_ph_flag"] = (df_fe["soil_pH"] < 5.5).astype(int)
# Crea un indice sintetico di stress ambientale: maggiore con molta umidità e pioggia, bassa temperatura
df_fe["env_stress_index"] = (df_fe["humidity"] * df_fe["rainfall"]) / (df_fe["temperature"] + 1)

# generatore di feature polinomiali di secondo grado (humidity², temperature², e humidity*temperature)
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_fe[["humidity", "temperature"]])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(["humidity", "temperature"]))
df_fe = pd.concat([df_fe.reset_index(drop=True), poly_df.iloc[:, 2:]], axis=1)

# Train/test split
X = df_fe.drop(columns=["disease_present"])
y = df_fe["disease_present"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest per classificare le piante sane/malate.
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
# Simulate climate change scenario
df_future = df_fe.copy()
df_future["temperature"] += 2
df_future["humidity"] += 10
X_future = df_future.drop(columns=["disease_present"]).loc[X_test.index]

# Predizione
original_preds = clf.predict_proba(X_test)[:, 1]
future_preds = clf.predict_proba(X_future)[:, 1]

# Plot risk shift heatmap
scenario_df = pd.DataFrame({
    "Original Risk": original_preds,
    "Future Risk (+2°C, +10% humidity)": future_preds
})

plt.figure(figsize=(8, 6))
sns.kdeplot(
    x=scenario_df["Original Risk"],
    y=scenario_df["Future Risk (+2°C, +10% humidity)"],
    fill=True, cmap="coolwarm", thresh=0.05
)
plt.title("Risk Score Distribution Shift: Climate Change Scenario")
plt.xlabel("Original Risk")
plt.ylabel("Future Risk")
plt.grid(True)
plt.tight_layout()
plt.show()

print(scenario_df)



#chatbot
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # Crea un'istanza del client OpenAI usando la chiave API
client = OpenAI(api_key=OPENAI_API_KEY)

print("Chiave", OPENAI_API_KEY)


df_global = None  # Variabile globale per il dataset

def load_data(file_path):
    global df_global
    df_global = pd.read_csv(file_path)

#SOLUZIONE 2 TOOLS MA PROBLEMI DI APIKEYS
# Crea un'istanza del client OpenAI usando la chiave API
client = OpenAI(api_key=OPENAI_API_KEY)

df_global = None  # Variabile globale per il dataset

df_global = pd.DataFrame({
    "Nome": ["PiantaA", "PiantaB", "PiantaC", "PiantaD", "PiantaE"],
    "temperature": [20, 22, 19, 24, 23],
    "humidity": [60, 65, 58, 70, 68],
    "rainfall": [100, 120, 110, 130, 125],
    "soil_pH": [6.5, 6.8, 6.4, 7.0, 6.9],
    "malattia": ["sana", "malata", "sana", "malata", "sana"]
})

# Funzioni che GPT può chiamare

df = pd.read_csv("plant_disease_dataset.csv")  # Assicurati che questo file esista

#  Media tra piante malate e sane
def media_malati_vs_sani():
    gruppi = df.groupby("malattia")
    medie = gruppi.mean(numeric_only=True)
    return medie.to_dict()

#  Variabili importanti con Random Forest
def variabili_importanti():
    X = pd.get_dummies(df.drop(columns=["malattia"]))
    y = df["malattia"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importanze = dict(zip(X.columns, model.feature_importances_))
    return dict(sorted(importanze.items(), key=lambda x: x[1], reverse=True))

#  Accuratezza modello di regressione logistica
def accuratezza_logistica():
    X = pd.get_dummies(df.drop(columns=["malattia"]))
    y = df["malattia"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuratezza": round(acc * 100, 2)}

#  Importanza della temperatura (correlazione)
def importanza_temperatura():
    if "temperatura" not in df.columns:
        return {"errore": "Colonna 'temperatura' non trovata nel dataset"}
    correlazione = df.corr(numeric_only=True)["malattia"].get("temperatura", None)
    return {"correlazione_temperatura_malattia": correlazione}

#  Simulazione +2°C
def rischio_con_aumento_temperatura():
    if "temperatura" not in df.columns:
        return {"errore": "Colonna 'temperatura' non trovata"}
    
    df_mod = df.copy()
    df_mod["temperatura"] += 2

    X_orig = pd.get_dummies(df.drop(columns=["malattia"]))
    X_mod = pd.get_dummies(df_mod.drop(columns=["malattia"]))
    X_orig, X_mod = X_orig.align(X_mod, join="left", axis=1, fill_value=0)
    y = df["malattia"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_orig, y)

    pred_orig = model.predict_proba(X_orig)[:, 1].mean()
    pred_mod = model.predict_proba(X_mod)[:, 1].mean()
    diff = pred_mod - pred_orig

    return {
        "rischio_originale": round(pred_orig, 3),
        "rischio_con_aumento_2C": round(pred_mod, 3),
        "aumento_rischio": round(diff, 3)
    }

#  Esecuzione delle funzioni GPT
def function_calling(tool_calls):
    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        match tool_call.function.name:
            case "media_malati_vs_sani":
                risultato = media_malati_vs_sani()
                return f"Medie tra piante sane e malate:\n{json.dumps(risultato, indent=2)}"
            case "variabili_importanti":
                risultato = variabili_importanti()
                return f"Variabili più influenti:\n{json.dumps(risultato, indent=2)}"
            case "accuratezza_logistica":
                risultato = accuratezza_logistica()
                return f"Accuratezza del modello di regressione logistica: {risultato['accuratezza']}%"
            case "importanza_temperatura":
                risultato = importanza_temperatura()
                return f"Correlazione temperatura-malattia:\n{json.dumps(risultato, indent=2)}"
            case "rischio_con_aumento_temperatura":
                risultato = rischio_con_aumento_temperatura()
                return (
                    f"Rischio malattia originale: {risultato['rischio_originale']}\n"
                    f"Rischio con +2°C: {risultato['rischio_con_aumento_2C']}\n"
                    f"Aumento stimato del rischio: {risultato['aumento_rischio']}"
                )

#  Tools disponibili per GPT
tools = [
    {
        "type": "function",
        "function": {
            "name": "media_malati_vs_sani",
            "description": "Calcola la media delle variabili tra piante sane e malate",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "variabili_importanti",
            "description": "Restituisce le variabili che influenzano maggiormente la malattia",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accuratezza_logistica",
            "description": "Calcola l'accuratezza del modello di regressione logistica",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "importanza_temperatura",
            "description": "Mostra quanto la temperatura è correlata con la malattia",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rischio_con_aumento_temperatura",
            "description": "Stima come cambia il rischio con un aumento della temperatura di 2°C",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

#  Interfaccia chat testuale
def chatbot():
    print("Chatbot attivo. Scrivi 'exit' per uscire.\n")
    while True:
        user_input = input("Domanda: ")
        if user_input.lower() == "exit":
            print("Arrivederci!")
            break

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # O anche: "gpt-3.5-turbo"
            messages=[{"role": "user", "content": user_input}],
            tools=tools,
            tool_choice="auto",
        )

        tool_calls = response.choices[0].message.tool_calls
        risposta = function_calling(tool_calls)
        print(risposta + "\n")

#  Avvia chatbot
if __name__ == "__main__":
    chatbot()


#soluzione semplice senza AI
# df_global = pd.DataFrame({
#     "Nome": ["PiantaA", "PiantaB", "PiantaC", "PiantaD", "PiantaE"],
#     "temperature": [20, 22, 19, 24, 23],
#     "humidity": [60, 65, 58, 70, 68],
#     "rainfall": [100, 120, 110, 130, 125],
#     "soil_pH": [6.5, 6.8, 6.4, 7.0, 6.9],
#     "malattia": ["sana", "malata", "sana", "malata", "sana"]
# })

# # === Analisi locali ===
# def analyze_data(analysis_type, column=None, group_by=None):
#     try:
#         df = df_global.copy()

#         if analysis_type == "mean":
#             if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
#                 result = df[column].mean()
#                 return f"Media di {column}: {result:.2f}"
#             return f"Colonna '{column}' non trovata o non numerica."

#         elif analysis_type == "group_by":
#             if group_by and column and group_by in df.columns and column in df.columns:
#                 if pd.api.types.is_numeric_dtype(df[column]):
#                     result = df.groupby(group_by)[column].mean().round(2)
#                     return f"Media di {column} per {group_by}:\n{result.to_string()}"
#                 return "La colonna per l'analisi deve essere numerica."
#             return "Specificare colonne valide per group_by."

#         return "Tipo di analisi non supportato."

#     except Exception as e:
#         return f"Errore nell'analisi: {str(e)}"

# # === Interprete della domanda ===
# def interpret_query(query):
#     q = query.lower()

#     if "importanza della temperatura" in q:
#         return analyze_data("mean", column="temperature")

#     elif "+2°c" in q or "più 2 gradi" in q or "riscaldamento climatico" in q:
#         try:
#             df = df_global.copy()
#             df["temperature_plus2"] = df["temperature"] + 2
#             media_rischio_normale = df["temperature"].mean()
#             media_rischio_plus2 = df["temperature_plus2"].mean()
#             diff = media_rischio_plus2 - media_rischio_normale
#             return (f"Con +2°C la temperatura media aumenta da {media_rischio_normale:.2f}°C "
#                     f"a {media_rischio_plus2:.2f}°C (+{diff:.2f}°C).")
#         except Exception as e:
#             return f"Errore nel calcolo rischio +2°C: {str(e)}"

#     elif "differenze medie tra piante malate e sane" in q:
#         return analyze_data("group_by", column="temperature", group_by="malattia")

#     elif "accuratezza del modello di regressione logistica" in q:
#         return "Non è ancora stato addestrato un modello. Integra un modello di regressione per calcolare l’accuratezza."

#     elif "variabili influenzano di più la malattia" in q:
#         return "Serve una regressione logistica o un modello di classificazione per stimare l'importanza delle variabili."

#     return ask_openai(query)

# # === Chiamata a OpenAI ===
# def ask_openai(query):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4",  # puoi anche usare "gpt-3.5-turbo"
#             messages=[
#                 {"role": "system", "content": "Sei un assistente esperto in analisi agronomiche."},
#                 {"role": "user", "content": query}
#             ]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"(ChatGPT): Errore chiamando OpenAI: {str(e)}"

# # === LOOP PRINCIPALE ===
# if __name__ == "__main__":
#     print("\nChatbot attivo. Scrivi 'exit' per uscire.\n")
#     while True:
#         user_query = input("Fai la tua domanda: ")
#         if user_query.lower() == "exit":
#             print("Arrivederci!")
#             break
#         risposta = interpret_query(user_query)
#         print(f"{risposta}\n")




#vorrei interrogare chat bot con queste domande :
# Qual è l'importanza della temperatura?
# Come cambia il rischio con +2°C?
# Quali sono le differenze medie tra piante malate e sane?
# “Qual è l’accuratezza del modello di regressione logistica?”

# “Quali variabili influenzano di più la malattia?”

# “Come cambia il rischio con il riscaldamento climatico?”



   





