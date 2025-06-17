import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
from collections import Counter

class DataManager:
    # Konstruktor
    def __init__(self, data_file):
        self.data_file=data_file
        self.n_pca = 0
           
    # Öffnen und Lesen einer Datei
    def load_csv_data(self, data_file="data\diabetes.csv", delimiter=','):
        try:
            df = pd.read_csv(data_file, delimiter=delimiter)
            df['PatientId'] = range(1, len(df) + 1)
            df.reset_index(drop=True, inplace=True)
            print(f"Daten erfolgreich geladen von {data_file}")
            return df
        except FileNotFoundError:
            print(f"Datei nicht gefunden: {data_file}")
        except pd.errors.ParserError:
            print(f"Fehler beim Lesen der Datei: {data_file}")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")
        return None
    
    # DataFrame DiabetesData standardisieren
    def data_to_standart(self,df):
        # Liste der Spalten, die NICHT standardisiert werden sollten
        print(df.columns)
        exclude_cols = ['PatientId','Z']
        df_excluded = df[exclude_cols]
        df_scaled = df.drop(columns=exclude_cols)
        df_scaled = (df_scaled - df_scaled.mean()) / df_scaled.std()
        df_result = pd.concat([df_excluded, df_scaled], axis=1)
        return df_result
    
    # Berechnet die Korrelation aller Merkmale mit dem Zielmerkmal und verwirft Merkmale, 
    # deren Korrelationskoeffizient kleiner als cc ist
    def data_to_correlation(self,df,cc=0.2):
        # Wir berechnen Korrelationen mit der Variable Z
        correlations = df.corr()['Z']
        # Entfernen Sie Z und PatientID aus den Korrelationen (falls vorhanden)
        correlations = correlations.drop(labels=['PatientId','Z'], errors='ignore')
        # Suchen Sie nach Features mit einer Korrelation unter dem Schwellenwert
        low_corr_features = correlations[correlations.abs() < cc].index.tolist()
        df_filtered = df.drop(columns=low_corr_features, errors='ignore')
        return correlations, df_filtered, low_corr_features
        
    # PCA Methode
    # Führt eine PCA-Zerlegung durch: berechnet die Kovarianzmatrix, die Eigenwerte und die Eigenvektoren.
    # Parameter:
    # - data: DataFrame — standardisierte Daten
    # - alpha: float — minimaler Eigenwert zur Auswahl der Komponenten (wenn method=0)
    # - c_var: float — Schwellenwert der kumulierten Varianz (wenn method=1)
    # - method: int — Methode zur Bestimmung der Anzahl der Hauptkomponenten (0: nach alpha, 1: nach kumulierter Varianz)
    # Gibt zurück:
    # - cov_matrix: die Kovarianzmatrix
    # - eigenvalues: absteigend sortierte Eigenwerte
    # - eigenvectors: entsprechende Eigenvektoren
    def data_to_PCA(self,data,value_method=1.0,n_method=0):
        # In Numpy-Array konvertieren
        if isinstance(data, pd.DataFrame):
            data_F = data.drop(columns=['PatientId','Z'], errors='ignore')
            col_names = data_F.columns
            data_PCA = data_F.values
        # Kovarianzmatrix
        cov_matrix = np.cov(data_PCA, rowvar=False)
        # Erstellen eines DataFrame zur Visualisierung
        cov_df = pd.DataFrame(cov_matrix, index=col_names, columns=col_names)
        # Eigenwerte und Eigenvektoren
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Nach absteigenden Eigenwerten sortieren
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Erstellen Sie eine Tabelle mit Eigenwerten und Varianzen
        sum_eigenvalues = np.sum(eigenvalues)
        # Der Varianzanteil jeder Komponente
        variance_ratio = eigenvalues / sum_eigenvalues
        # Kumulierte Varianz
        cumulative_variance = np.cumsum(variance_ratio)
        # Erstellen einer Tabelle
        pca_method = pd.DataFrame({
            'Eigenwert': np.round(eigenvalues, 4),
            'Varianzanteil': np.round(variance_ratio, 4),
            'Kumulierte Varianz': np.round(cumulative_variance, 4)
        })
        if n_method == 0:
            self.n_pca = (pca_method['Eigenwert'] > value_method).sum()
        else:
            self.n_pca = (pca_method['Kumulierte Varianz'] > value_method).idxmax()

        # Wir berechnen die Werte der Hauptkomponenten
        X = data[col_names].values 
        principal_components = np.dot(X, eigenvectors)

        # Erstellen Sie einen DataFrame mit den Ergebnissen
        pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        pca_df = pd.DataFrame(principal_components, columns=pca_columns)
        pca_df[['Z']] = data[['Z']].reset_index(drop=True)
        pca_df[['PatientId']] = data[['PatientId']].reset_index(drop=True)
        
        # Entfernen zusätzlicher PC-Spalten
        cols_to_drop = [f'PC{i+1}' for i in range(pca_df.shape[1]) if f'PC{i+1}' in pca_df.columns and i > self.n_pca]
        pca_df = pca_df.drop(columns=cols_to_drop)
        return pca_df, cov_df, pca_method
    
    # Kohonen Methode
    # Führt die Konstruktion einer Kohonen-Karte mit anschließender Clusterung durch
    # Parameter:
    # - data: DataFrame — PCA Daten
    # - n_map: int — Abmessungen der Kohonen-Karte
    # - n_clusters: int — Anzahl der Cluster
    # Gibt zurück:
    # - result_df: DataFrame - anfängliche Datentabelle mit hinzugefügten Koordinaten auf der Kohonen-Karte und Clusternummer
    # - kohonen_map_df: DataFrame Карта Кохонена
    def data_to_SOM(self, data, n_map=9, n_clusters=5):
    # Datentransformation: 'PatientId','Z' trennen und nur numerische Merkmale berücksichtigen
        if isinstance(data, pd.DataFrame):
            features = data.drop(columns=['PatientId','Z']).values
            Z_values = data['Z'].values
            PatientId_values = data['PatientId'].values
        else:
            raise ValueError("Es wurde ein pandas.DataFrame mit Spalte 'Z' erwartet.")

        # SOM-Schulung
        som = MiniSom(x=n_map, y=n_map, input_len=features.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(features)
        som.train(features, 10000)

        # BMU-Koordinaten abrufen
        bmu_coords = np.array([som.winner(x) for x in features])
        bmu_x, bmu_y = bmu_coords[:, 0], bmu_coords[:, 1]
        bmu_idx = np.ravel_multi_index(bmu_coords.T, (n_map, n_map))

        # Gruppieren nach Kartenknoten und Berechnen von Z
        bmu_data = pd.DataFrame({'BMU_x': bmu_x, 'BMU_y': bmu_y, 'Z': Z_values})
        kohonen_map_df = bmu_data.groupby(['BMU_x', 'BMU_y'])['Z'].agg(
            Count='count',
            Z0_count=lambda x: (x == 0).sum(),
            Z1_count=lambda x: (x == 1).sum()
        ).reset_index()

        # Abrufen von Knotengewichten und Clustering von Kartenknoten
        weights = som.get_weights().reshape(-1, features.shape[1])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        node_labels = kmeans.fit_predict(weights)

        # Wir weisen jeder Beobachtung basierend auf ihrer BMU einen Cluster zu
        cluster_labels = np.array([node_labels[idx] for idx in bmu_idx])

        # DataFrame mit Knotenkoordinaten und Clustern
        node_coords = [(i, j) for i in range(n_map) for j in range(n_map)]
        node_df = pd.DataFrame(node_coords, columns=['BMU_x', 'BMU_y'])
        node_df['cluster'] = node_labels
        for i in range(weights.shape[1]):
            node_df[f'weight_{i}'] = weights[:, i]

        # Kombinieren Sie Knotenstatistiken mit Clusterkoordinaten und Beschriftungen
        kohonen_map_df = kohonen_map_df.merge(node_df, on=['BMU_x', 'BMU_y'], how='left')

        # Wir sammeln das Endergebnis
        result_df = data.copy()
        result_df['BMU_x'] = bmu_x
        result_df['BMU_y'] = bmu_y
        result_df['BMU_idx'] = bmu_idx
        result_df['Cluster_on_SOM'] = cluster_labels
        result_df['Z'] = Z_values
        result_df['PatientId'] = PatientId_values
        return result_df, kohonen_map_df
    
    # Wir sammeln das Endergebnis
    def data_to_result(self,proc_df, SOM_df):
        # Wir bilden zusammenfassende Tabellen
        merged = pd.merge(proc_df, SOM_df, on='PatientId', how='inner')
        print(merged.columns)
        result_df = merged[['PatientId','Z_x','F1','F2','F3','F4','F5','F6','F7','F8', 'PC1','PC2','PC3','PC4','Cluster_on_SOM']]
        result_df = result_df.rename(columns={
            'Z_x': 'Z',
            'F1': 'Pregnancies',
            'F2': 'Glucose',
            'F3': 'BloodPressure',
            'F4': 'SkinThickness',
            'F5': 'Insulin',
            'F6': 'BMI',
            'F7': 'DiabetesPedigreeFunction',
            'F8': 'Age'
        })
        z_df = result_df.groupby('Cluster_on_SOM')['Z'].value_counts().unstack(fill_value=0)
        z_df.index.name = 'Cluster'
        z_df = z_df.rename(columns={
            0: 'Gesund',
            1: 'Krank'
        })
        # Fügen Sie eine neue Spalte mit Risikobewertung hinzu
        total = z_df['Gesund'] + z_df['Krank']
        share_krank = z_df['Krank'] / total

        z_df['Gesundheitsgruppe'] = share_krank.apply(
            lambda x:   'Gruppe Gesundheit' if x < 1/3 else (
                        'Gruppe mit erhöhtem Risiko' if x > 2/3 else
                        'Gruppe mit Risiko')
        )
        z_df['_highlight'] = share_krank.apply(
            lambda x:   False if x < 1/3 else True
            )
                
        return result_df, z_df