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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']

    if file and allowed_file(file.filename):
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, na_values=["--", "na", "NaN", "HURLEY"])
        elif file.filename.endswith('.json'):
            df = load_json_to_dataframe(file)

        before_rows, before_columns = df.shape
        missing_values = df.isnull().sum()

        df_numeric = df.select_dtypes(include=[np.number])
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (df_numeric < lower_bound) | (df_numeric > upper_bound)
        outlier_count = outliers.sum()

        duplicates = df[df.duplicated(keep=False)]
        duplicates_count = len(duplicates) // 2

        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                df[column] = df[column].replace(0, np.nan)
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                df[column] = df[column].apply(lambda x: median_value if x < lower_bound[column] or x > upper_bound[column] else x)
            elif df[column].dtype == 'object':
                def is_numeric_string(val):
                    try:
                        float(val)
                        return True
                    except:
                        return False

                mask_numeric_in_text = df[column].apply(lambda x: is_numeric_string(x) if pd.notnull(x) else False)
                if mask_numeric_in_text.any():
                    mode_value = df[column][~mask_numeric_in_text].mode()[0] if not df[column][~mask_numeric_in_text].mode().empty else "Inconnu"
                    df.loc[mask_numeric_in_text, column] = mode_value

                mode_value = df[column].mode()[0] if not df[column].mode().empty else "Inconnu"
                df[column] = df[column].fillna(mode_value)

        rows_to_remove = (outliers.sum(axis=1) > 2) | (df.isnull().sum(axis=1) > 2)
        df_cleaned = df[~rows_to_remove]
        df_cleaned = df_cleaned.drop_duplicates(keep='first')

        after_rows, after_columns = df_cleaned.shape

        
        cleaned_filename = 'abdou_' + os.path.splitext(file.filename)[0] + '.csv'

        cleaned_filepath = os.path.join(CLEANED_FOLDER, cleaned_filename)
        df_cleaned.to_csv(cleaned_filepath, index=False)

        preview_data = df_cleaned.head(50).to_dict(orient='records')
        missing_values_dict = missing_values.to_dict()
        outlier_count_dict = outlier_count.to_dict()

        return render_template('resultat.html',
                               before_rows=before_rows,
                               before_columns=before_columns,
                               after_rows=after_rows,
                               after_columns=after_columns,
                               missing_values=missing_values_dict,
                               outlier_count=outlier_count_dict,
                               duplicates_count=duplicates_count,
                               preview_data=preview_data,
                               cleaned_filename=cleaned_filename)

    return 'Erreur lors du téléchargement du fichier', 400

# Petit test pour GitHub

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(CLEANED_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
