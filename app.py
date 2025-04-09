from flask import Flask, request, render_template, send_from_directory
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'data'
CLEANED_FOLDER = 'cleaned_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_json_to_dataframe(file):
    data = json.load(file)
    return pd.json_normalize(data)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, na_values=["--", "na", "NaN", "HURLEY", "__", " "])
        elif file.filename.endswith('.json'):
            df = load_json_to_dataframe(file)

        before_rows, before_columns = df.shape
        missing_values = df.isnull().sum()

        # Conversion forcée des colonnes numériques mal typées
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                continue

        # Détection des valeurs aberrantes pour les colonnes numériques
        df_numeric = df.select_dtypes(include=[np.number])
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df_numeric < lower_bound) | (df_numeric > upper_bound)
        outlier_count = outliers.sum()

        # Doublons
        duplicates = df[df.duplicated(keep=False)]
        duplicates_count = len(duplicates) // 2

        # Traitement des valeurs aberrantes : Remplacer par NaN pour les numériques
        for col in df_numeric.columns:
            df[col] = df[col].where(~outliers[col], np.nan)

        # Gestion des colonnes texte contenant des chiffres
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Inconnu"
            df[col] = df[col].fillna(mode_val)

        # Traitement des valeurs manquantes
        total_na_rows = df.isnull().sum(axis=1)
        if (total_na_rows > 0).sum() <= 3:
            df.dropna(inplace=True)
        else:
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].replace(0, np.nan)
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                elif df[col].dtype == 'object':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Inconnu"
                    df[col].fillna(mode_val, inplace=True)

        # Suppression des lignes avec trop de valeurs manquantes ou aberrantes
        rows_to_remove = (df.isnull().sum(axis=1) > 2) | (outliers.sum(axis=1) > 2)
        df_cleaned = df[~rows_to_remove]
        df_cleaned = df_cleaned.drop_duplicates(keep='first')

        after_rows, after_columns = df_cleaned.shape

        # Normalisation
        df_normalized = df_cleaned.copy()
        for col in df_normalized.select_dtypes(include=[np.number]).columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val != min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)

        # Sauvegarde
        cleaned_filename = 'abdou_' + os.path.splitext(file.filename)[0] + '.csv'
        cleaned_filepath = os.path.join(CLEANED_FOLDER, cleaned_filename)
        df_cleaned.to_csv(cleaned_filepath, index=False)

        normalized_filename = 'normalized_' + os.path.splitext(file.filename)[0] + '.csv'
        normalized_filepath = os.path.join(CLEANED_FOLDER, normalized_filename)
        df_normalized.to_csv(normalized_filepath, index=False)

        preview_data = df_cleaned.head(30).to_dict(orient='records')
        normalized_data = df_normalized.head(30).to_dict(orient='records')

        return render_template('resultat.html',
                               before_rows=before_rows,
                               before_columns=before_columns,
                               after_rows=after_rows,
                               after_columns=after_columns,
                               missing_values=missing_values.to_dict(),
                               outlier_count=outlier_count.to_dict(),
                               duplicates_count=duplicates_count,
                               preview_data=preview_data,
                               normalized_data=normalized_data,
                               cleaned_filename=cleaned_filename,
                               normalized_filename=normalized_filename)

    return 'Erreur lors du téléchargement du fichier', 400

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(CLEANED_FOLDER, filename, as_attachment=True)

@app.route('/download_normalized_file/<filename>')
def download_normalized_file(filename):
    return send_from_directory(CLEANED_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
