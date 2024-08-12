import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression



# Carregar les dades


def load_data():
    caract = pd.read_csv('caracteritzacio_dades.txt', sep='\t', encoding='utf-16')
    estab = pd.read_csv('estabilitat_dades.txt', sep='\t', encoding='utf-16')
    est = pd.read_csv('estadistiques_dades.txt', sep='\t', encoding='utf-16')
    matriu = pd.read_csv('matriu_dades.txt', sep='\t', encoding='utf-16')
    oleo = pd.read_csv('olea_dades.txt', sep='\t', encoding='utf-16')
    
    return oleo, matriu, est, caract, estab

    

oleo, matriu, est, caract, estab = load_data()

def check_password():
    if 'password_entered' not in st.session_state:
        # First run, show input for password
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        def set_password():
            if st.session_state["password"] == "Additius":  # Replace with your actual password
                st.session_state.password_entered = True
            else:
                st.session_state.password_entered = False
                st.error("Password incorrecta")

        st.text_input("Introdueix la contrasenya:", type="password", key="password", on_change=set_password)
        return False
    else:
        return True

if check_password():
    
    
    # Títol de l'aplicació
    st.title("Anàlisi d'Estabilitat de Xantofil·les en Oleoresines")

    # Selecció de l'usuari per a la secció a visualitzar
    option = st.sidebar.selectbox(
        "Selecciona una opció",
        ["Disseny Experimental", "Dades Disponibles", "Anàlisi Descriptiva", "Anàlisi de Correlacions", "Modelització", "Calculadora"]
    )

    if option == "Disseny Experimental":
       

        
        st.write("""
        #### Objectiu de l'Estudi
        L'objectiu d'aquest estudi és analitzar la relació entre la composició de la matèria primera (oleoresines) i l'estabilitat de les xantofil·les en el producte final. S'ha observat que l'estabilitat del producte final varia considerablement en funció de la oleoresina utilitzada, però no se sap per què. Actualment, només es coneix el contingut de xantofil·les de la oleoresina en el moment de la seva arribada, però la resta de la composició és desconeguda.
         
        ## Disseny Experimental

        ### Mostres
        - 50 oleoresines diferents, cadascuna amb dues reaccions pilot per minimitzar la variabilitat.
        - Cada reacció pilot inclou un previ i un final, característics del procés industrial. Les potencials variables target son: 'Previ 7', 'Final 7', 'Previ 42', 'Final 42'

        ### Condicions de la Reacció Pilot
        - El previ conté només sílice.
        - El final conté sílice i carbonat.

        ### Condicions de l'Assaig d'Estabilitat
        - Les mostres es guarden en una càmera climàtica a 40 graus Celsius i 75% d'humitat durant 42 dies (7 setmanes).
        - Es realitza una anàlisi de les xantofil·les totals cada setmana per determinar el percentatge de retenció de xantofil·les.

        ### Dades Addicionals Recollides
        - Mesures de clorofil·les, polifenols, compostos volàtils (tres grups: productes de degradació oxidativa d'àcids grassos i carotenoides, i productes de degradació termooxidativa dels carotenoides), acidesa (àcids grassos lliures), composició d'àcids grassos (saturats, monosaturats i poliinsaturats), tocoferols (quatre tipus), ferro i coure.
        - Paràmetres de la reacció: pèrdua de xantofil·les durant l'escalfament, rendiment de la reacció, temperatura màxima del reactor, temps de canvi de color en la saponificació, control de qualitat (color, densitat, pH, granulometria i humitat).

        ### Metriques d'Avaluació
        Per avaluar els models utilitzats en l'estudi, es van utilitzar diverses mètriques, incloent:
        - Mean Squared Error (MSE): Mesura la mitjana dels errors al quadrat entre les prediccions i els valors reals.
        - R squared (R²): Coeficient de determinació, que mesura la proporció de variabilitat en la variable dependent explicada per les variables independents en el model.

        ## Selecció del Millor Model i Variables Rellevants

        ### Explicació Teòrica de la Selecció del Millor Model
        Per seleccionar el millor model per predir la variable dependent (es pot seleccionar entre qualsevol de les disponibles: 'Previ 7', 'Final 7', 'Previ 42', 'Final 42'), es va seguir una metodologia sistemàtica que inclou la selecció de models candidats, la definició de paràmetres, la divisió de les dades en conjunts d'entrenament i prova, i l'avaluació de les prestacions dels models.

        #### Preprocessament de Dades:
        - Es converteixen totes les dades a format numèric per manejar qualsevol error de coerció.
        - S'identifiquen i seleccionen les variables numèriques i categòriques per al preprocessor.
        - S'utilitza estratègies d'imputació per substituir els valors nuls amb la mitjana per les variables numèriques i OneHotEncoder per les variables categòriques.

        #### Divisió de les Dades:
        - Es divideixen les dades en conjunts d'entrenament i prova en una proporció 80/20 per assegurar la validació creuada.

        #### Selecció i Entrenament de Models:
        - Es consideren diversos models de regressió, incloent-hi la regressió lineal, el Random Forest, i el Gradient Boosting.
        - S'utilitza GridSearchCV per ajustar els hiperparàmetres dels models amb l'objectiu de maximitzar el coeficient de determinació (R²).
        - S'entrenen els models amb les dades d'entrenament i es s'avaluen amb les dades de prova.

        #### Avaluació de Models:
        - Es calcula els scores de R² per comparar els models.
        - Es selecciona els millors paràmetres per a cada model basant-se en el seu rendiment durant la validació creuada.
        - El model Random Forest obte el millor score R², sent seleccionat com el millor model.

        ### Selecció de Variables Rellevants
        Per seleccionar les variables rellevants, es s'utilitzar la importància de les característiques del model Random Forest.

        #### Càlcul de la Importància de les Característiques:
        - Es calculen les importàncies de les característiques del model Random Forest.
        - S'ordena la importància de cada característica per identificar les més significatives.

      
        """) 
    elif option == "Dades Disponibles":
        st.header("Dades Disponibles")
        dataset_name = st.selectbox("Selecciona el conjunt de dades", ["Oleo", "Matriu", "Estadístiques", "Caracterització", "Estabilitat"])
    
        if dataset_name == "Oleo":
            st.write("## Dades de l'Oleo")
            st.write(oleo.head(10))
        
        elif dataset_name == "Matriu":
            st.write("## Dades de la Matriu")
            st.write(matriu.head(10))
        
        elif dataset_name == "Estadístiques":
            st.write("## Dades d'Estadístiques")
            st.write(est.head())
        
        elif dataset_name == "Caracterització":
            st.write("## Dades de Caracterització")
            st.write(caract.head(10))
        
        else:
            st.write("## Dades d'Estabilitat")
            st.write(estab.head(10))

    elif option == "Anàlisi Descriptiva":
        st.header("Anàlisi Descriptiva")
        dataset_name = st.selectbox("Selecciona el conjunt de dades per a l'anàlisi descriptiva", ["Oleo", "Matriu", "Estadístiques", "Caracterització", "Estabilitat"])


        data = None
        if dataset_name == "Oleo":
            data = oleo
        elif dataset_name == "Matriu":
            data = matriu
        elif dataset_name == "Estadístiques":
            data = est
        elif dataset_name == "Caracterització":
            data = caract
        else:
            data = estab
    
        st.write(f"## Anàlisi Descriptiva de {dataset_name}")
        st.write(data.describe())
    
        # Visualitzacions
        st.write("### Distribució de les Variables")
    
        numeric_columns = data.select_dtypes(include=['number']).columns
        num_columns = len(numeric_columns)
        cols_per_row = 3

        for i in range(0, num_columns, cols_per_row):
            row_columns = numeric_columns[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
        
            for col, column in zip(cols, row_columns):
                with col:
                    st.write(f"#### Distribució de {column}")
                    fig, ax = plt.subplots()
                
                    try:
                        ax.hist(data[column].dropna(), bins=30, edgecolor='k')
                        ax.set_title(f'Distribució de column')
                        ax.set_xlabel(column)
                        ax.set_ylabel('Freqüència')
                       
                    except np.linalg.LinAlgError:
                       ax.hist(data[column].dropna(), bins=30, edgecolor='k')
                       ax.set_title(f'Distribució de column')
                       ax.set_xlabel(column)
                       ax.set_ylabel('Freqüència')
                    
                    st.pyplot(fig)
                    
                    
       

    elif option == "Anàlisi de Correlacions":
        st.header("Anàlisi de Correlacions")
       
        y_options = ['Previ 7',  'Previ 42',  'Lot ASM', 'Lot', 'Productor', 'NR']
        
        est_numeric = est.apply(pd.to_numeric, errors='coerce')
        est_numeric = est_numeric.drop(columns=y_options)
        
        
        # Calcula la matriu de correlacions
        corr_matrix = est_numeric.corr()
        # corr_matrix = corr_matrix.drop(['Previ 7'])
        # corr_matrix = corr_matrix.drop(['Previ 42'])
        # Filtrar per les correlacions de Final 7 i Final 42
        corr_final7 = corr_matrix[['Final 7']].drop(['Final 7'])
        corr_final42 = corr_matrix[['Final 42']].drop(['Final 42'])
    
        corr_min, corr_max = st.slider("Rang de correlació", 0.0, 1.0, (0.4, 0.8), 0.05)
    
    
        # Crear el heatmap
        st.subheader("Matriu de Correlacions")
            
        col1, col2 = st.columns(2)

        with col1:
            st.write("Correlació entre les variables i Final 7")
            filtered_corr_final7 = corr_final7[(corr_final7['Final 7'].abs() >= corr_min) & (corr_final7['Final 7'].abs() <= corr_max)]
            fig1, ax1 = plt.subplots(figsize=(10, 30))
            cax1 = ax1.matshow(filtered_corr_final7, cmap="YlOrRd")
            fig1.colorbar(cax1)
            ax1.set_yticks(range(len(filtered_corr_final7.index)))
            ax1.set_yticklabels(filtered_corr_final7.index, fontsize=20)
            st.pyplot(fig1)

        with col2:
            st.write("Correlació entre les variables i Final 42")
            filtered_corr_final42 = corr_final42[(corr_final42['Final 42'].abs() >= corr_min) & (corr_final42['Final 42'].abs() <= corr_max)]
            fig2, ax2 = plt.subplots(figsize=(10, 30))
            cax2 = ax2.matshow(filtered_corr_final42, cmap="YlOrRd")
            fig2.colorbar(cax2)
            ax2.set_yticks(range(len(filtered_corr_final42.index)))
            ax2.set_yticklabels(filtered_corr_final42.index, fontsize=20)
            st.pyplot(fig2)

        st.subheader("Resum de Correlacions Significatives")

        # Resum de les correlacions més altes per Final 7 i Final 42
        top_corr_final7 = corr_final7['Final 7'].sort_values(ascending=False).head(10)
        top_corr_final42 = corr_final42['Final 42'].sort_values(ascending=False).head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Correlacions amb 'Final 7'")
            st.table(top_corr_final7)

        with col2:
            st.write("Correlacions amb 'Final 42'")
            st.table(top_corr_final42)
        
        
          
        


    elif option == "Modelització":
    
        y_options = ['Previ 7', 'Final 7', 'Previ 42', 'Final 42']
        selected_y = st.selectbox("Selecciona la variable Y:", y_options, index=3)
        
        y_options = ['Previ 7', 'Final 7', 'Previ 42', 'Final 42', 'Lot ASM', 'Lot', 'NR']
        
        st.header("Modelització")
     
        
        est = pd.merge(est, matriu[['NR', 'Origen', 'Extracció']], on='NR', how='left')

        est_numeric = est.apply(pd.to_numeric, errors='coerce')
        X = est_numeric.drop(columns=y_options)
        y = est_numeric[selected_y]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), X.columns)
            ]
        )
 
        X_preprocessed = preprocessor.fit_transform(X)
      
        feature_names = X.columns.tolist()

        # Seleccionar les 10 característiques més rellevants
        selector = SelectKBest(score_func=f_regression, k=10)
        X_selected = selector.fit_transform(X_preprocessed, y)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # Ordenar les característiques seleccionades per rellevància (F-value)
        f_values = selector.scores_
        selected_features_with_f = sorted(
            zip(selected_feature_names, f_values[selected_feature_indices]),
            key=lambda x: x[1],
            reverse=True
        )
        ordered_feature_names = [feature for feature, _ in selected_features_with_f]

        # Mostrar les característiques seleccionades i els seus valors F
        st.write("## Variables més rellevants")
        st.write("""
            ### Selecció de Variables Rellevants

            En la modelització predictiva, és essencial identificar quines variables (característiques) tenen un impacte significatiu en la variable objectiu. Això es fa mitjançant l'anàlisi de la regressió F (ANOVA) i els valors p associats.

            #### Càlcul del Valor F

            El valor F en una anàlisi de regressió compara la variància explicada per cada variable independent amb la variància no explicada (o error). Un valor F alt indica que una gran part de la variància de la variable objectiu és explicada per la variable independent.

            - **Alt valor F**: La variable independent té una gran capacitat d'explicar la variància de la variable dependent.
            - **Baix valor F**: La variable independent té poca capacitat d'explicar la variància de la variable dependent.

            #### Valor P Associat

            El valor p mesura la probabilitat que la relació observada entre les variables independent i dependent sigui deguda a l'atzar. Un valor p baix indica que és poc probable que la relació sigui per atzar, suggerint una relació significativa.

            - **Valor p baix (per exemple, < 0.05)**: La variable independent és estadísticament significativa.
            - **Valor p alt (per exemple, > 0.05)**: La variable independent pot no ser significativa.

            #### Procediment de Selecció

            1. **Càlcul de valors F i p per a cada variable independent (X)**.
            2. **Selecció de les variables més rellevants** basant-se en els valors F més alts i els valors p més baixos. Per fer-ho, utilitzem la llibreria `SelectKBest` del paquet `sklearn.feature_selection` per seleccionar les millors variables per al model. `SelectKBest` selecciona les k millors característiques basant-se en els tests estadístics, en aquest cas, la regressió F.
            3. **Construcció del model final** utilitzant les variables seleccionades.
            """)

        f_values_df = pd.DataFrame({
            'Feature': ordered_feature_names,
            'F-Value': [f for _, f in selected_features_with_f]
        }).sort_values(by='F-Value', ascending=False)
        st.table(f_values_df)

        # Dividir les dades en entrenament i prova
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Entrenar el model amb les característiques seleccionades
        rf_model = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=10, min_samples_leaf=4, random_state=42)
        rf_model.fit(X_train, y_train)

        # Rendiment del model
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write("### Interpretació dels Resultats del Model ")
        st.write("""
        
        - **MSE (Error Quadràtic Mitjà)**: Aquest valor indica, de mitjana, quant s'equivoca el model en les seves prediccions. Un valor més baix és millor, ja que significa que les prediccions estan més a prop dels valors reals.
        - **R² (Coeficient de Determinació)**: Aquest valor mostra quina part de la variabilitat de les dades és explicada pel model. Un valor proper a 1 indica un model molt bo, mentre que un valor baix indica que el model no explica bé les dades.

        **Com interpretar els resultats:**

        - **Bon Ajust:** Si el model té un MSE baix i un R² alt, significa que és bo per predir l'estabilitat de les xantofil·les.
        - **Marge de Millora:** Encara que el model sigui bo, sempre es pot millorar ajustant paràmetres o afegint més dades.
        """)
       
  
        results_df = pd.DataFrame([{
            'Model': 'Random Forest',
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2
        }])
        st.write("## Resultats de la Modelització")
        st.write(results_df)
    
    elif option == "Calculadora":
    
        st.header("Calculadora")
    
        y_options = ['Previ 7', 'Final 7', 'Previ 42', 'Final 42']
        selected_y = st.selectbox("Selecciona la variable Y:", y_options, index=3)
        y_options = ['Previ 7', 'Final 7', 'Previ 42', 'Final 42', 'Lot ASM', 'Lot', 'Productor', 'NR']
        
        
        est = pd.merge(est, matriu[['NR', 'Origen', 'Extracció']], on='NR', how='left')

        est_numeric = est.apply(pd.to_numeric, errors='coerce')
        X = est_numeric.drop(columns=y_options)
        y = est_numeric[selected_y]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), X.columns)
            ]
        )

        X_preprocessed = preprocessor.fit_transform(X)
        feature_names = X.columns.tolist()

        # Seleccionar les 10 característiques més rellevants
        selector = SelectKBest(score_func=f_regression, k=10)
        X_selected = selector.fit_transform(X_preprocessed, y)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_feature_indices]

        # Ordenar les característiques seleccionades per rellevància (F-value)
        f_values = selector.scores_
        selected_features_with_f = sorted(
            zip(selected_feature_names, f_values[selected_feature_indices]),
            key=lambda x: x[1],
            reverse=True
        )
        ordered_feature_names = [feature for feature, _ in selected_features_with_f]

        

        f_values_df = pd.DataFrame({
            'Feature': ordered_feature_names,
            'F-Value': [f for _, f in selected_features_with_f]
        }).sort_values(by='F-Value', ascending=False)
       

        # Dividir les dades en entrenament i prova
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Entrenar el model amb les característiques seleccionades
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=42)
        rf_model.fit(X_train, y_train)

        # Rendiment del model
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        st.write("### Metriques del Model ")
       
       
  
        results_df = pd.DataFrame([{
            'Model': 'Random Forest',
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2
        }])
       
        st.write(results_df)
        

        # Configurar el número de columnas por fila
        num_columns = 3  # Puedes ajustar este valor según sea necesario
        num_rows = (len(ordered_feature_names) + num_columns - 1) // num_columns

        input_values = {}

        for row in range(num_rows):
            cols = st.columns(num_columns)
            for i, col in enumerate(cols):
                index = row * num_columns + i
                if index < len(ordered_feature_names):
                    feature = ordered_feature_names[index]
                    input_values[feature] = col.number_input(feature)

   
        
        # Preparar les dades d'entrada com a DataFrame per a la predicció
        input_data = pd.DataFrame([input_values])

        # Assegurar-se que totes les columnes necessàries estan presents en input_data
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Omplir amb zero o algun altre valor predeterminat

        # Aplicar el preprocessor a les dades d'entrada
        input_data_preprocessed = preprocessor.transform(input_data)

        # Seleccionar les característiques rellevants
        input_data_selected = selector.transform(input_data_preprocessed)
        
        # Realitzar la predicció
        prediction = rf_model.predict(input_data_selected)
        
        # Mostrar el resultat de la predicció
        st.subheader(f"Predicció per '{selected_y}': {prediction[0]:.4f}")
