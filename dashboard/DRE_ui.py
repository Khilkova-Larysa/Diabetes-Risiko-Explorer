import streamlit as st
import pandas as pd
import numpy as np
from src.DRE_data import DataManager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class UI:
    def __init__(self):
        self.manager = DataManager("data/diabetes.csv")
        self.pages = {
            "Main": self.page_main,
            "Daten": self.page_data,
            "Daten herunterladen": self.page_download,
            "Datenstandardisierung": self.page_standardization, 
            "Korrelationsanalyse": self.page_correlation, 
            "Hauptkomponentenanalyse": self.page_principal, 
            "Clusteranalyse: Kohonen-Methode + K-Mean-Methode": self.page_cluster_kohonen,
            "Interpretation der Ergebnisse": self.page_interpretation
        }
           
    def run(self):
        # Seitenkopf und Grundkonfiguration
        st.set_page_config(
            page_title="Diabetes-Risiko-Explorer",
            layout="wide",
            page_icon=chr(0x1F9EC),
            initial_sidebar_state="expanded"
        )
        st.sidebar.markdown(
            '<span style="color: darkred; font-weight: bold;font-size: 20px;">SECTION:</span>',
            unsafe_allow_html=True
        )
        page = st.sidebar.radio("", [
            "Main",
            "Daten",
            "Daten herunterladen",
            "Datenstandardisierung",
            "Korrelationsanalyse", 
            "Hauptkomponentenanalyse", 
            "Clusteranalyse: Kohonen-Methode + K-Mean-Methode",
            "Interpretation der Ergebnisse"
        ], index = 0)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        st.sidebar.image("images/logo_schnecke_2.png", use_container_width=True)
        self.pages[page]()

    # html - Tabellendruckformat für Klastermethode
    def df_to_html(self, df) -> str:
        # Manuelle HTML-Generierung mit Linienfärbung
        rows_html = ""
        for idx, row in df.iterrows():
            bg_color = "darkred" if row['_highlight'] else "white"
            ft_color = "white" if row['_highlight'] else "black"
            row_html = f"<tr style='background-color: {bg_color}; color: {ft_color};'>"
            row_html += f"<td><strong>{idx}</strong></td>"
            for val in row.drop('_highlight'):
                row_html += f"<td>{val}</td>"
            row_html += "</tr>"
            rows_html += row_html

        # Schlagzeilen
        columns = df.drop(columns=['_highlight']).columns
        columns_html = "".join([f"<th>{col}</th>" for col in columns])
        columns_html = f"<th>{df.index.name if df.index.name else ''}</th>" + columns_html

        # CSS-Styling hinzufügen
        html = f"""
<style>
    .table-container {{
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }}
    table {{
        width: 60%;
        border-collapse: collapse;
        font-size: 14px;
    }}
    thead th {{
        text-align: center !important;
        font-size: 16px;
        color: darkred;
        background-color: #e6e6e6;
    }}
    tbody td {{
        text-align: left !important;
        font-size: 14px;
    }}
</style>
<div class="table-container">
    <table>
        <thead><tr>{columns_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
"""
        return html
    
    # Wir zeichnen eine Spalte mit Koordinaten. Wir verwenden Kohonen-Karten zum Zeichnen, 
    # da matplotlib.bar3D die Sortierreihenfolge nicht beibehält.
    def draw_custom_bar(self, ax, x, y, z, dx, dy, dz, color):
        xx = [x, x+dx]
        yy = [y, y+dy]
        zz = [z, z+dz]
    
        vertices = [
            [(xx[0], yy[0], zz[0]), (xx[1], yy[0], zz[0]), (xx[1], yy[1], zz[0]), (xx[0], yy[1], zz[0])],  # низ
            [(xx[0], yy[0], zz[1]), (xx[1], yy[0], zz[1]), (xx[1], yy[1], zz[1]), (xx[0], yy[1], zz[1])],  # верх
            [(xx[0], yy[0], zz[0]), (xx[1], yy[0], zz[0]), (xx[1], yy[0], zz[1]), (xx[0], yy[0], zz[1])],  # перед
            [(xx[0], yy[1], zz[0]), (xx[1], yy[1], zz[0]), (xx[1], yy[1], zz[1]), (xx[0], yy[1], zz[1])],  # зад
            [(xx[0], yy[0], zz[0]), (xx[0], yy[1], zz[0]), (xx[0], yy[1], zz[1]), (xx[0], yy[0], zz[1])],  # лев
            [(xx[1], yy[0], zz[0]), (xx[1], yy[1], zz[0]), (xx[1], yy[1], zz[1]), (xx[1], yy[0], zz[1])],  # прав
        ]

        poly = Poly3DCollection(vertices, facecolors=color, edgecolors='k', alpha=0.8)
        ax.add_collection3d(poly)

    # Main page
    def page_main(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: black;">Willkommen bei der App '
            '<span style="color: darkred;">\"Diabetes-Risiko-Explorer\"</span>!</h1>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
                - {chr(128105)}‍{chr(128187)} **Entwicklerin:** Larysa Khilkova https://www.linkedin.com/in/dr-larysa-khilkova-b09128269/ 
                - {chr(128197)} **Datum:** Juni 2025  
                - {chr(128197)} **Technologien:** 
                    - **Plattform:** Python
                    - **Datenverarbeitung:** Pandas, Numpy
                    - **Visualisierung:** Matplotlib, Seaborn  
                    - **Schnittstelle:** Streamlit
                - {chr(9997)} **Projektbeschreibung:** Typ-2-Diabetes zählt zu den häufigsten und heimtückischsten 
                Erkrankungen unserer Zeit. Ziel dieses Projekts ist es, mithilfe von Datenanalyse und Clusterverfahren 
                die Einflussfaktoren und Mechanismen der Krankheitsentwicklung besser zu verstehen, um wirksame 
                Präventionsstrategien zu entwickeln.
                - 🔢**Daten:** Die Daten von 768 Frauen, die von der Website zur Verfügung gestellt wurden, wurden analysiert. 
                https://www.kaggle.com/datasets/sefakocakalay/diabets 
            """)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
                <span style="color: darkred;font-weight: bold;">Wählen Sie einen Abschnitt aus dem Menü auf der linken Seite aus, um zu beginnen.</span>
            """, unsafe_allow_html=True)
    
    # Daten page
    def page_data(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">DATEN</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
                In diesem Abschnitt werden die verwendeten Datensätze vorgestellt. Hier finden Sie Informationen zur  
                Struktur und Qualität der Daten, die die Grundlage für die Analyse und Modellierung bilden.
            """)
        st.markdown("""
<h1 style="font-size: 20px; color: darkred;">Datenelementen:</h1>
                    
- **$\mathcal{F}_{1}$ Pregnancies:**
    - *Beschreibung*: Anzahl der Schwangerschaften bei der Patientin.
    - *Datentyp*: Ganze Zahl.
    - *Kontext*: Eine Schwangerschaft kann ein Risikofaktor für die Entwicklung von Typ-2-Diabetes sein.
- **$\mathcal{F}_{2}$ Glucose:**
    - *Beschreibung*: Blutzuckerspiegel (in mg/dl) nach einem 2-Stunden-Glukosetoleranztest.
    - *Datentyp*: Gleitkommazahl.
    - *Kontext*: Einer der wichtigsten Risikofaktoren für Diabetes.
- **$\mathcal{F}_{3}$ BloodPressure:**
    - *Beschreibung*: Diastolischer Blutdruck (in mmHg).
    - *Datentyp*: Ganze Zahl.
    - *Kontext*: Hoher Blutdruck kann mit metabolischen Störungen zusammenhängen.
- **$\mathcal{F}_{4}$ SkinThickness:**
    - *Beschreibung*: Dicke der Hautfalte am Trizeps (in mm).
    - *Datentyp*: Ganze Zahl.
    - *Kontext*: Wird als indirektes Maß für Körperfett verwendet.
- **$\mathcal{F}_{5}$ Insulin:**
    - *Beschreibung*: Insulinspiegel im Blutserum (in μU/ml).
    - *Datentyp*: Ganze Zahl.
    - *Kontext*: Kann auf Insulinresistenz hinweisen - ein wichtiger Faktor bei Diabetes.
- **$\mathcal{F}_{6}$ BMI (Body Mass Index):**
    - *Beschreibung*: Körpermasseindex (Gewicht/Größe²), kg/m².
    - *Datentyp*: Gleitkommazahl.
    - *Kontext*: Übergewicht erhöht das Diabetesrisiko.
- **$\mathcal{F}_{7}$ DiabetesPedigreeFunction:**
    - *Beschreibung*: Familiärer Risiko-Faktor (Schätzung des Diabetesrisikos basierend auf familiärer Vorgeschichte).
    - *Datentyp*: Gleitkommazahl.
    - *Kontext*: Je höher der Wert, desto wahrscheinlicher eine genetische Veranlagung.
- **$\mathcal{F}_{8}$ Age:**
    - *Beschreibung*: Alter der Patientin (in Jahren).
    - *Datentyp*: Ganze Zahl.
    - *Kontext*: Das Risiko für Diabetes steigt mit dem Alter.
- **$\mathcal{Z}$ Diagnoseinformationen** (0:  nein, 1: ja) <span style="color:blue;">(Zweck der Studie)</span>.
        """, unsafe_allow_html=True)
    
    # Daten herunterladen page
    def page_download(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Daten herunterladen</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
                In diesem Abschnitt erfolgt das Laden der Daten aus der Datei **„diabetes.csv“**. 
                Die Daten werden in einem DataFrame gespeichert, um sie weiter analysieren zu können.
                    
                **Ziel** dieses Abschnitts ist es, die Daten für die nachfolgenden Schritte der Analyse oder Visualisierung vorzubereiten.
            """)
        # Initialisierung session_state
        if 'data_ready' not in st.session_state:
            st.session_state.data_ready = False

        if st.button("📂 Daten laden"):
            df_DD = self.manager.load_csv_data()
            if df_DD is not None:
                df_DD_1 = df_DD.rename(columns={
                    "Pregnancies": "F1",
                    "Glucose": "F2",
                    "BloodPressure": "F3",
                    "SkinThickness": "F4",
                    "Insulin": "F5",
                    "BMI": "F6",
                    "DiabetesPedigreeFunction": "F7",
                    "Age": "F8",
                    "Outcome": "Z"
                })
                st.session_state.df_DD_1 = df_DD_1
                st.session_state.data_ready = True
                df_DD_1.to_csv("data/processed/procdiabetes_data.csv", index=False, encoding="utf-8")
            else:
                st.error("Fehler beim Laden der Daten.")
        if st.session_state.data_ready:
            df_DD_1 = st.session_state.df_DD_1
            st.success("Daten erfolgreich geladen!")
            st.dataframe(df_DD_1.head())
            
    # Datenstandardisierung page
    def page_standardization(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Datenstandartisierung</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
**Ziel** dieses Abschnitts ist es, die Datenelemente auf eine einheitliche Skala zu bringen, um die 
Leistung von Machine-Learning-Algorithmen zu verbessern, die empfindlich auf die Verteilung und den Maßstab 
der Merkmale reagieren (z. B. lineare Regression, Hauptkomponentenanalyse (PCA) und das k-Means-Clustering-Verfahren).
	
Die Standardisierung erfolgt nach der Formel:
        """)
        st.latex(r"""
\mathcal{F}'= \frac{\mathcal{F}-\mu}{\sigma}
        """)
        st.markdown("""
wobei $\mu$ der Mittelwert und $\sigma$ die Standardabweichung des Merkmals ist.
	
Durch die Standardisierung haben alle Merkmale einen Mittelwert $\mu'=0$ von 0 und eine Standardabweichung $\sigma=1$, 
was zu einer stabileren und effizienteren Arbeitsweise der Datenanalysealgorithmen beiträgt.
        """)
        # Initialisierung session_state
        if 'standardized_data_ready' not in st.session_state:
            st.session_state.standardized_data_ready = False

        if st.button("🧊 Datenstandartisierung"):
            df = self.manager.load_csv_data("data/processed/procdiabetes_data.csv")

            if df is not None:
                df_SD = self.manager.data_to_standart(df)
                if df_SD is not None:
                    st.session_state.df = df
                    st.session_state.df_SD = df_SD
                    st.session_state.standardized_data_ready = True
                    df_SD.to_csv("data/processed/standardized_data.csv", index=False)
            else:
                st.error("Daten werden nicht geladen, gehen Sie zur Registerkarte 'Daten herunterladen' und laden Sie die Daten herunter")   
        # Diagramme
        if st.session_state.standardized_data_ready:
            st.success("Daten erfolgreich standartisieren!")
            df = st.session_state.df
            df_SD = st.session_state.df_SD
            st.dataframe(df_SD.head())
            st.markdown(
                        '<h1 style="font-size: 20px; color: darkred;">Vergleich von Original- und standardisierten Daten</h1>',
                        unsafe_allow_html=True
                    )
            # Achsen auswählen
            column_names = [col for col in df.columns if col != "Z"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="font-weight:bold; margin-bottom: 2px;">Wählen Achsen X:</div>', unsafe_allow_html=True)
                x_axis = st.selectbox("", options=column_names, index=0)

            with col2:
                st.markdown('<div style="font-weight:bold; margin-bottom: 2px;">Wählen Achsen Y:</div>', unsafe_allow_html=True)
                y_axis = st.selectbox("", options=column_names, index=1)
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # Das erste Diagramm zeigt die Originaldaten
            ax1.scatter(df[x_axis], df[y_axis], alpha=0.7, color='blue',s=5)
            ax1.set_xlabel(x_axis)
            ax1.set_ylabel(y_axis)
            ax1.set_title("Original")
            ax1.set_aspect('equal')
            
            # Das zweite Diagramm zeigt standardisierte Daten
            ax2.scatter(df_SD[x_axis], df_SD[y_axis], alpha=0.7, color='orange',s=5)
            ax2.set_xlabel(x_axis)
            ax2.set_ylabel(y_axis)
            ax2.set_title("Standardisiert")
            ax2.set_aspect('equal')
            st.pyplot(fig)
            fig.savefig('plots/original_standartisiert.png', dpi=300, bbox_inches='tight')

            st.markdown('''**Das Ergebnis der Datenstandardisierung** ist ein transformierter Merkmalsdatensatz, bei dem gilt:
- **Der Mittelwert** ($\mu$) jedes Merkmals ist gleich 0;
- **Die Standardabweichung** ($\sigma$) jedes Merkmals ist gleich 1.

Das bedeutet, dass alle Merkmale auf denselben Maßstab gebracht wurden, ohne Verschiebungen oder Unterschiede im Maßstab.
                    ''')
    
    # Korrelationsanalyse page
    def page_correlation(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Korrelationsanalyse</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
**Die Korrelationsanalyse** ermöglicht es, festzustellen, ob statistische Zusammenhänge zwischen Merkmalen bestehen 
und wie stark diese ausgeprägt sind.
                    
In diesem Abschnitt untersuchen wir den Einfluss einzelner Merkmale auf das Ergebnis – das Vorhandensein oder 
Nichtvorhandensein von Diabetes bei den Patienten.
        """)
        # Initialisierung session_state
        if 'correlation_ready' not in st.session_state:
            st.session_state.correlation_ready = False
        
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown("""
|Korrelationskoeffizient| Korrelationsstärke                               |
|------------------------|-------------------------------------------------|
| 0,80 - 1,00            | Sehr starker bzw. nahezu perfekter Zusammenhang |
| 0,60 - 0,79            | Starker linearer Zusammenhang                   |
| 0,40 - 0,59            | Mäßiger Zusammenhang                            |
| 0,20 - 0,39            | Schwacher, aber erkennbarer Zusammenhang        |
| 0,00 - 0,19            | Kaum ein Zusammenhang                           |
""")
           
        with col2:
            st.markdown('<div style="font-weight:bold; margin-bottom: 2px;">Wählen Sie die niedrigere Korrelationsstufe:</div>', 
                    unsafe_allow_html=True)
            n_cor = float(st.selectbox("", options=["0.8", "0.6", "0.4", "0.2", "0.1", "0.05", "0"], index=5))
        
        if st.button("📈 Korrelation"):
            df_SD = self.manager.load_csv_data("data\processed\standardized_data.csv")
            if df_SD is not None:
                correlations, df_CD, low_corr_features = self.manager.data_to_correlation(df_SD,n_cor)
                st.session_state.correlations = correlations
                st.session_state.df_CD = df_CD
                st.session_state.low_corr_features = low_corr_features
                st.session_state.correlation_ready = True
                df_CD.to_csv("data/processed/correlation_data.csv", index=False)
            else:
                st.error("Die Daten werden nicht geladen. Gehen Sie zur Registerkarte „Datenstandartisierung“, um die Originaldaten für diese Registerkarte zu erhalten.")   
        # Ergebnisse
        if st.session_state.correlation_ready:
            correlations = st.session_state.correlations
            df_CD = st.session_state.df_CD
            low_corr_features = st.session_state.low_corr_features
            # Wir geben eine Linie mit Korrelationskoeffizienten aus
            colored_parts = []
            for col, val in correlations.items():
                color = 'blue' if abs(val) >= n_cor else 'red'
                colored_parts.append(f'<span style="color:{color}">({col}: {val:.3f})</span>')
            # Fassen wir alles in einer Zeile zusammen
            output_str = ', '.join(colored_parts)
            st.markdown("**Korrelation** zwischen Merkmalen und Zielindikator:")
            st.markdown(output_str, unsafe_allow_html=True)
            st.markdown(f"""
| Korrelationskoeffizient | Korrelationsstärke                                           |
|-------------------------|--------------------------------------------------------------|
| Pregnancies             | $r_1=$ {correlations.loc["F1"]:.3f} - <span style="color:orange">schwacher Zusammenhang</span>  |
| Glucose                 | $r_2=$ {correlations.loc["F2"]:.3f} - <span style="color:green">mäßiger Zusammenhang</span>    |
| BloodPressure           | $r_3=$ {correlations.loc["F3"]:.3f} - <span style="color:red">kaum ein Zusammenhang</span>   |
| SkinThickness           | $r_4=$ {correlations.loc["F4"]:.3f} - <span style="color:red">kaum ein Zusammenhang</span>   |
| Insulin                 | $r_5=$ {correlations.loc["F5"]:.3f} - <span style="color:red">kaum ein Zusammenhang</span>   |
| BMI                     | $r_6=$ {correlations.loc["F6"]:.3f} - <span style="color:orange">schwacher Zusammenhang</span>  |
| DiabetesPedigreeFunction| $r_7=$ {correlations.loc["F7"]:.3f} - <span style="color:red">kaum ein Zusammenhang</span>   |
| Age                     | $r_8=$ {correlations.loc["F8"]:.3f} - <span style="color:orange">schwacher Zusammenhang</span>  |
                """, unsafe_allow_html=True)   
   
    # Hauptkomponentenanalyse page
    def page_principal(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Hauptkomponentenanalyse<br>(Principal Component Analysis, PCA)</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
<h1 style="font-size: 20px; color: darkred;">Ziel der PCA:</h1>
Die Anzahl der Merkmale (Variablen) zu reduzieren, indem nur diejenigen beibehalten werden, die die größte Varianz in den Daten 
erklären, wobei der Informationsverlust so gering wie möglich gehalten wird.
                """, unsafe_allow_html=True)
        st.markdown("""
<h1 style="font-size: 20px; color: darkred;">Grundidee:</h1>
                    
**PCA** sucht neue Achsen (Hauptkomponenten), die:
- lineare Kombinationen der ursprünglichen Merkmale sind;
- unabhängig (orthogonal) voneinander sind;
- nach absteigender Varianz geordnet sind (die erste Komponente hat die höchste Varianz, die zweite die höchste Varianz
    unter der Bedingung der Orthogonalität zur ersten usw.).
                """, unsafe_allow_html=True)
               
        # Initialisierung session_state
        if 'PCA_ready' not in st.session_state:
            st.session_state.PCA_ready = False
        st.markdown('<div style="font-weight:bold; margin-bottom: 2px;">Wählen Sie die Methode und Schwellenwert zur Bestimmung der Anzahl der Hauptkomponenten:</div>', 
                    unsafe_allow_html=True)
        col1, col2 = st.columns([3,1])
        with col1:
            n_method = int(st.selectbox("0: nach Eigenwerte (Schwellenwert=1.0), 1: nach kumulierter Varianz (Schwellenwert=75%)", options=["0", "1"], index=0))
            default_value = 0.75 if n_method == 1 else 1.0
        with col2:
            value_method = st.number_input("Schwellenwert:", value=default_value)
            
        if st.button("🌀 PCA"):
            df_CD = self.manager.load_csv_data("data/processed/correlation_data.csv")
            if df_CD is not None:
                pca_df, cov_df, pca_method = self.manager.data_to_PCA(df_CD,value_method,n_method)
                st.session_state.pca_df = pca_df
                st.session_state.cov_df = cov_df
                st.session_state.n_pca = self.manager.n_pca
                st.session_state.pca_method = pca_method
                st.session_state.PCA_ready = True
                pca_df.to_csv("data/processed/PCA_data.csv", index=False)
            else:
                st.error("Die Daten werden nicht geladen. Gehen Sie zur Registerkarte „Korrelationsanalyse“, um die Originaldaten für diese Registerkarte zu erhalten.")   
        
        # Diagramme
        if st.session_state.PCA_ready:
            pca_df = st.session_state.pca_df
            cov_df = st.session_state.cov_df
            pca_method = st.session_state.pca_method
            n_pca = st.session_state.n_pca
                       
            # Visualisierung der Kovarianzmatrix
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(cov_df, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar=True, ax=ax1)
            ax1.set_title("Kovarianzmatrix")
            fig1.tight_layout()
            st.pyplot(fig1)
            fig1.savefig('plots/Kovarianzmatrix.png', dpi=300, bbox_inches='tight')

            # Wir bestimmen die Anzahl der Hauptkomponenten
            st.markdown("""**Eigenwerte der Kovarianzmatrix und ihr Beitrag zur Gesamtvarianz**
                        """)
            st.markdown(f"""Wir wahlen {n_pca} Hauptkomponenten
                        """)
            pca_method['_highlight'] = (pca_method['Eigenwert']>= value_method) if n_method == 0 else (pca_method['Kumulierte Varianz'] < value_method)
            st.markdown(self.df_to_html(pca_method), unsafe_allow_html=True)
            
            # Zeichnen des Kettel-Diagramms
            components = np.arange(1, len(pca_method) + 1)  # Komponenten-Nummern für die X-Achse
            fig2, ax2 = plt.subplots(figsize=(10, 6))  # Höhe hier reduziert для лучшей пропорции
            ax2.plot(components, pca_method['Eigenwert'], marker='o', label='Eigenwert')
            ax2.scatter(components[:n_pca], pca_method['Eigenwert'][:n_pca], color='red', zorder=n_pca, label=f'Top {n_pca} Komponenten')
            ax2.set_title("Scree-Plot (Methode von Cattell)", fontsize=14)
            ax2.set_xlabel("Komponenten-Nummer", fontsize=12)
            ax2.set_ylabel("Eigenwert", fontsize=12)
            ax2.set_xticks(components)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2)
            fig2.savefig('plots/Kettel_Diagramms.png', dpi=300, bbox_inches='tight')

            # Verteilung der Patienten im Hauptkomponentenraum
            st.markdown("""**Verteilung der Patienten im Hauptkomponentenraum**
                        """)
            fig3, axes = plt.subplots(int(n_pca/2), 2, figsize=(12, 2.5*n_pca))
            pairs = [(1, j) for j in range(2, n_pca + 1)] # Liste der zu erstellenden Komponentenindizes
            legend_elements = [
                    Patch(facecolor='purple', edgecolor='purple', label='Diabetiker'),
                    Patch(facecolor='yellow', edgecolor='yellow', label='Nicht-Diabetiker')
                ]

            for ax3, (i, j) in zip(axes.flatten(), pairs):
                ax3.scatter(pca_df[f'PC{i}'], pca_df[f'PC{j}'], c=pca_df['Z'], cmap='viridis', alpha=0.7)
                ax3.set_xlabel(f'PC{i}')
                ax3.set_ylabel(f'PC{j}')
                ax3.set_title(f'PC{i} vs PC{j}')
                ax3.legend(handles=legend_elements, loc='best')

            fig3.tight_layout()
            st.pyplot(fig3)
            fig3.savefig('plots/Hauptkomponentenraum.png', dpi=300, bbox_inches='tight')

    # Clusteranalyse: Kohonen-Methode page
    def page_cluster_kohonen(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Clusteranalyse: Kohonen-Methode<br>(Self-Organizing Map, SOM)</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
<span style="color:darkred;">**Die Kohonen-Methode**</span> projiziert hochdimensionale Eingabedaten auf ein zwei- (oder ein-) dimensionales Gitter 
der Kohonen-Karte, sodass ähnliche Eingabedaten in benachbarte Bereiche der Karte abgebildet werden. 
			
Dies ermöglicht die Visualisierung, Clusterbildung und Analyse der Datenstruktur.
- <span style="color:darkred;">**Am Eingang**</span> haben wir hochdimensionale Daten.
- <span style="color:darkred;">**Am Ausgang**</span> eine Kohonen-Karte, die die Datenstruktur visualisier.
            """, unsafe_allow_html=True)
                
        # Initialisierung session_state
        if 'SOM_ready' not in st.session_state:
            st.session_state.SOM_ready = False
        st.markdown('<div style="font-weight:bold; margin-bottom: 2px;">Festlegen der Kohonen-Kartengröße und der Anzahl der Cluster:</div>', 
                    unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])
        with col1:
            n_map = st.number_input("Kohonen-Kartengröße:", value=7)
        with col2:
            n_clusters = st.number_input("Anzahl der Cluster:", value=4)
            
        if st.button("🗺️ Self-Organizing Map"):
            pca_df = self.manager.load_csv_data("data/processed/PCA_data.csv")
            if pca_df is not None:
                som_df, kohonen_map_df = self.manager.data_to_SOM(pca_df, n_map, n_clusters)
                st.session_state.som_df = som_df
                st.session_state.kohonen_map_df = kohonen_map_df
                st.session_state.SOM_ready = True
                st.session_state.n_pca = pca_df.shape[1]-1
                som_df.to_csv("data/processed/SOM_data.csv", index=False)
                kohonen_map_df.to_csv("data/processed/Kohonen_map_data.csv", index=False)
            else:
                st.error("Die Daten werden nicht geladen. Gehen Sie zur Registerkarte „Korrelationsanalyse“, um die Originaldaten für diese Registerkarte zu erhalten.")   
        # Diagramme
        if st.session_state.SOM_ready:
            n_pca = st.session_state.n_pca
            som_df = st.session_state.som_df
            kohonen_map_df = st.session_state.kohonen_map_df
                        
            # Visualisierung der Karte Kohonen
            # Vorbereiten einer Farbkarte für Cluster
            unique_clusters = np.sort(kohonen_map_df['cluster'].dropna().unique())
            palette1 = [
                "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#ffe119", 
                "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
            ]
            palette2 = [
                "#f4a3b3", "#a6e9a4", "#a6b8f4", "#f9bd91", "#fff7a8", 
                "#cfa6dd", "#aafafa", "#f7b3f1", "#e4f9a5", "#fde2de"
            ]
            palette1 = palette1[:len(unique_clusters)]
            cluster_color_map_1 = dict(zip(unique_clusters, palette1))

            palette2 = palette2[:len(unique_clusters)]
            cluster_color_map_2 = dict(zip(unique_clusters, palette2))

            fig1, ax1 = plt.subplots(figsize=(10, 8))
            cell_size = 1

            # Quadrate zeichnen
            for _, row in kohonen_map_df.iterrows():
                x, y = row['BMU_x'], row['BMU_y']
                cluster = row['cluster']
                count = int(row['Count']) if not pd.isna(row['Count']) else 0
                z1 = int(row['Z1_count']) if 'Z1_count' in row and not pd.isna(row['Z1_count']) else 0
                z0 = int(row['Z0_count']) if 'Z0_count' in row and not pd.isna(row['Z0_count']) else 0

                facecolor = cluster_color_map_2.get(cluster, 'lightgray')

                rect = patches.Rectangle(
                    (x, y),
                    cell_size, cell_size,
                    facecolor=facecolor,
                    edgecolor='black',
                    linewidth=1
                )
                ax1.add_patch(rect)

                # Text innerhalb der Zelle
                text = f"{count}: {z1} / {z0}"
                ax1.text(
                    x + 0.5, y + 0.5,
                    text,
                    ha='center', va='center',
                    fontsize=8,
                    color='black'
                )

            # Anpassen von Achsen und Darstellung
            ax1.set_xlim(0, n_map)
            ax1.set_ylim(0, n_map)
            ax1.set_xticks(range(n_map))
            ax1.set_yticks(range(n_map))
            ax1.set_aspect('equal')
            ax1.invert_yaxis()
            ax1.set_title('Die Kohonen-Karte (Verteilung der Patienten)\n (Gesamtzahl: krank / gesund)')
            ax1.set_xlabel('BMU (Best Matching Unit) x')
            ax1.set_ylabel('BMU_y')
            legend_patches = [
                patches.Patch(color=color, label=f'Claster {int(cl)}')
                for cl, color in cluster_color_map_2.items()
            ]
            ax1.legend(
                handles=legend_patches,
                title='Clusters',
                loc='upper center',
                bbox_to_anchor=(0.5, -0.08),
                ncol=len(legend_patches),
                frameon=False
            )
            # Zeichnen und Speichern der Zeichnung
            plt.tight_layout()
            st.pyplot(fig1)
            fig1.savefig('plots/Karte_Kohonen_2D.png', dpi=300, bbox_inches='tight')

            if n_pca is not None:
                # Verteilung der Patienten im Hauptkomponentenraum
                st.markdown(f"""**Verteilung der Clasters im {n_pca}D-Hauptkomponentenraum**
                            """)
                st.markdown("""
                           Jede Zelle der Kohonen-Karte hat ihre eigenen Koordinaten im Raum der Hauptkomponenten. Deren Verteilung stellen wir in einer Grafik dar.
                            """)
                fig2, ax2 = plt.subplots(int(n_pca/2), 2, figsize=(12, 2.5*n_pca))
                pairs = [(0, j) for j in range(1, n_pca-1)]
                
                legend_patches = [
                                    patches.Patch(color=color, label=f'Claster {int(cl)}')
                                    for cl, color in cluster_color_map_1.items()
                                ]
                for ax2, (i, j) in zip(ax2.flatten(), pairs):
                    colors = kohonen_map_df['cluster'].map(cluster_color_map_1)
                    ax2.scatter(
                            kohonen_map_df[f'weight_{i}'], 
                            kohonen_map_df[f'weight_{j}'], 
                            c=colors, 
                            alpha=0.7
                        )
                    ax2.set_xlabel(f'weight_{i}')
                    ax2.set_ylabel(f'weight_{j}')
                    ax2.set_title(f'weight_{i} vs weight_{j}')
                    ax2.legend(handles=legend_patches, loc='best')

                fig2.tight_layout()
                st.pyplot(fig2)
                fig2.savefig('plots/Kohonen_Hauptkomponentenraum.png', dpi=300, bbox_inches='tight')
            else:
                st.info("Führen Sie zunächst auf der entsprechenden Registerkarte eine Hauptkomponentenanalyse durch.")
            
            # Datenstruktur für gesunde und kranke Patienten
            st.markdown("""
                            **Die Kohonen-Karte** ermöglicht es, **die Struktur der Daten** zu erkennen.
                        """)
            fig3 = plt.figure(figsize=(20, 10))
            ax3 = fig3.add_subplot(1, 2, 1, projection='3d') # für gesunde Patienten
            ax4 = fig3.add_subplot(1, 2, 2, projection='3d') # für kranke Patienten

            # Graph für Z=0 (gesunde Patienten)
            for y in range(n_map-1,-1,-1):
                for x in range(n_map):   
                    row = kohonen_map_df[(kohonen_map_df['BMU_y'] == y) & (kohonen_map_df['BMU_x'] == x)]
                    if not row.empty:
                        z = 0
                        dx = dy = 0.8
                        dz = row['Z0_count'].iloc[0]
                        cluster = row['cluster'].iloc[0]
                        color = cluster_color_map_1[cluster]
                        self.draw_custom_bar(ax3, x, y, z, dx, dy, dz, color)
            
            ax3.view_init(elev=20, azim=45) # Изменение угла камеры, помогло при прорисовке
            ax3.set_xlabel('BMU_x')
            ax3.set_ylabel('BMU_y')
            ax3.set_zlabel('Count')
            ax3.set_title('Kohonens Karte für gesunde Patienten')

            # Graph für Z=1 (kranke Patienten)
            for y in range(n_map-1,-1,-1): 
                for x in range(n_map):
                    row = kohonen_map_df[(kohonen_map_df['BMU_y'] == y) & (kohonen_map_df['BMU_x'] == x)]
                    if not row.empty:
                        z = 0
                        dx = 0.8
                        dz = row['Z1_count'].iloc[0]
                        cluster = row['cluster'].iloc[0]
                        color = cluster_color_map_1[cluster]
                        self.draw_custom_bar(ax4, x, y, z, dx, dy, dz, color)
            
            ax4.view_init(elev=20, azim=45) # Изменение угла камеры, помогло при прорисовке
            ax4.set_xlabel('BMU_x')
            ax4.set_ylabel('BMU_y')
            ax4.set_zlabel('Count')
            ax4.set_title('Kohonens Karte für kranke Patienten')

            patches_legend = [patches.Patch(color=cluster_color_map_1[c], label=f'Cluster {c}') for c in unique_clusters]
            fig3.legend(handles=patches_legend, loc='upper center', ncol=len(unique_clusters), bbox_to_anchor=(0.5, 0.1))
            st.pyplot(fig3)
            fig3.savefig('plots/Karte_Kohonen_3D.png', dpi=300, bbox_inches='tight')
    
    # Interpretation der Ergebnisse page
    def page_interpretation(self):
        st.markdown(
            '<h1 style="font-size: 33px; color: darkred;">Interpretation der Ergebnisse</h1>',
            unsafe_allow_html=True
            )
        st.markdown("""
Wir analysieren die Zusammensetzung der Cluster in Bezug auf das Vorhandensein von Typ-2-Diabetes 
bei den Patienten, die dem jeweiligen Cluster zugeordnet wurden, sowie die Werte der Datenmerkmale 
für jedes Objekt im Cluster.
                """, unsafe_allow_html=True)
        st.markdown('<h2 style="font-size: 25px; color: darkred;">Zusammensetzung der Cluster</h2>',unsafe_allow_html=True)
        
         # Initialisierung session_state
        if 'result_ready' not in st.session_state:
            st.session_state.result_ready = False
                    
        if st.button("🔍 Analysieren"):
            proc_df = self.manager.load_csv_data("data/processed/procdiabetes_data.csv")
            som_df = self.manager.load_csv_data("data/processed/SOM_data.csv")
            if proc_df is not None:
                if som_df is not None:
                    result_df, z_df = self.manager.data_to_result(proc_df,som_df)
                    st.session_state.result_df = result_df
                    st.session_state.z_df = z_df
                    st.session_state.result_ready = True
                    result_df.to_csv("data/processed/result_data.csv", index=False)
                else:
                    st.error("Die Daten werden nicht geladen. Gehen Sie zur Registerkarte „Clusteranalyse: Kohonen-Methode + K-Mean-Methode“, um die Originaldaten für diese Registerkarte zu erhalten.")   
            else:
                st.error("Die Daten werden nicht geladen. Gehen Sie zur Registerkarte „Daten herunterladen“, um die Originaldaten für diese Registerkarte zu erhalten.")   
        # Ergebnisse
        if st.session_state.result_ready:
            result_df = st.session_state.result_df
            z_df = st.session_state.z_df
            st.markdown("Zusammensetzung der Cluster")
            st.markdown(self.df_to_html(z_df), unsafe_allow_html=True)       
       
            # Verteilung der Patienten in Cluster
            n_claster = result_df['Cluster_on_SOM'].nunique()

            palette1 = [
                "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#ffe119", 
                "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
                ]
            st.markdown(f"""**Verteilung der Patienten in {n_claster} Cluster**
                            """)
        
            # Merkmalspaare
            pairs = [
                    ('Pregnancies', 'Glucose'),
                    ('BloodPressure', 'SkinThickness'),
                    ('Insulin', 'BMI'),
                    ('DiabetesPedigreeFunction', 'Age')
            ]

            # Einrichten von Diagrammen
            sns.set(style='whitegrid')
            # Создаем одну фигуру с 4 подграфиками
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 2 строки, 2 столбца
            axes = axes.flatten()  # Упрощаем доступ: axes[0], axes[1], axes[2], axes[3]

            # Zeichnen Sie jeden Teilgraphen
            for i, (x_feat, y_feat) in enumerate(pairs):
                ax = axes[i]
                # Grenzen
                x_min, x_max = result_df[x_feat].min(), result_df[x_feat].max()
                y_min, y_max = result_df[y_feat].min(), result_df[y_feat].max()
                x_1_3 = x_min + (x_max - x_min) / 3
                x_2_3 = x_min + 2 * (x_max - x_min) / 3
                y_1_3 = y_min + (y_max - y_min) / 3
                y_2_3 = y_min + 2 * (y_max - y_min) / 3
                # Punkte
                sns.scatterplot(
                    data=result_df,
                    x=x_feat,
                    y=y_feat,
                    hue='Cluster_on_SOM',
                    palette=palette1,
                    legend=False, 
                    ax=ax
                )

                # Linie
                ax.axvline(x_1_3, color='green', linestyle='--', label='1/3 (X)')
                ax.axvline(x_2_3, color='red', linestyle='--', label='2/3 (X)')
                ax.axhline(y_1_3, color='green', linestyle='--', label='1/3 (Y)')
                ax.axhline(y_2_3, color='red', linestyle='--', label='2/3 (Y)')

                ax.set_title(f'{x_feat} vs {y_feat}')
                ax.set_xlabel(x_feat)
                ax.set_ylabel(y_feat)

            # Wir nehmen die Legende einmal heraus
            patches_legend = [patches.Patch(color=palette1[c], label=f'Cluster {c}') for c in range(n_claster)]
            fig.legend(handles=patches_legend, loc='upper center', ncol=n_claster, bbox_to_anchor=(0.5, 0))

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            st.pyplot(fig)
            fig.savefig('plots/Cluster_2x2.png', dpi=300, bbox_inches='tight')