
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



# Carregar les dades
@st.cache_resource
def load_data():
    caract = pd.read_excel('caracteritzacio_dades.xlsx')
    estab = pd.read_excel('estabilitat_dades.xlsx')
    est = pd.read_excel('estadistiques_dades.xlsx')
    matriu = pd.read_excel('matriu_dades.xlsx')
    oleo = pd.read_excel('olea_dades.xlsx')

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

        # Calcula la matriu de correlacions
        corr_matrix = est.corr()
        corr_matrix = corr_matrix.drop(['Previ 7'])
        corr_matrix = corr_matrix.drop(['Previ 42'])
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
    
        est_numeric = est.apply(pd.to_numeric, errors='coerce')
        X = est_numeric.drop(columns=y_options)
        y = est_numeric[selected_y]
        
        feature_names = X.columns.tolist()
        
        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='mean'), X.columns)
                ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
    
        best_params_rf = {
            'n_estimators': 300,
            'max_depth': 30,
            'min_samples_split': 10,
            'min_samples_leaf': 4
        }
        # Initialize and train the Random Forest model with the best parameters
        rf_model = RandomForestRegressor(
            n_estimators=best_params_rf['n_estimators'],
            max_depth=best_params_rf['max_depth'],
            min_samples_split=best_params_rf['min_samples_split'],
            min_samples_leaf=best_params_rf['min_samples_leaf'],
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Display results
        results_df = pd.DataFrame([{
            'Model': 'Random Forest',
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R²': train_r2,
            'Test R²': test_r2
        }])
        st.write("## Resultats de la Modelització")
        st.write(results_df)
        st.write("##### Prediccions vs. Valors Reals")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Valors Reals')
        ax.set_ylabel('Prediccions')
        ax.set_title('Random Forest Predictions vs. Actual')
        st.pyplot(fig)

        # Feature Importance

        importances = rf_model.feature_importances_
        total_importance = np.sum(importances)
        indices = np.argsort(importances)[::-1]

        # Show only top 10 features
        st.write("### Top 10 Característiques")
        for f in range(min(10, len(feature_names))):
            feature_name = feature_names[indices[f]]
            importance_value = importances[indices[f]]
            importance_percentage = (importance_value / total_importance) * 100
            st.write(f"{feature_name}: {importance_percentage:.2f}%")
        



        st.write("")
    elif option == "Calculadora":
        y_options = ['Previ 7', 'Final 7', 'Previ 42', 'Final 42']
        selected_y = st.selectbox("Selecciona la variable Y:", y_options, index=3)

        est_numeric = est.apply(pd.to_numeric, errors='coerce')
        X = est_numeric.drop(columns=y_options)
        y = est_numeric[selected_y]

        # Guardem els noms de les característiques per a l'anàlisi posterior
        feature_names = X.columns.tolist()

        # Eliminar les columnes que tenen tots els valors mancants
        X = X.dropna(axis=1, how='all')

        # Guardem els nous noms de les característiques després d'eliminar les columnes completament mancants
        feature_names = X.columns.tolist()

        # Data preprocessing pipeline
        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply the preprocessing pipeline to the training and testing data
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)

        # Ensure the number of features after preprocessing matches the original feature names
        if X_train.shape[1] != len(feature_names):
            # Log the difference in features
            original_features = set(feature_names)
            preprocessed_features = set([f"feature_{i}" for i in range(X_train.shape[1])])
            missing_features = original_features - preprocessed_features
            extra_features = preprocessed_features - original_features
            raise ValueError(
                f"Number of features after preprocessing ({X_train.shape[1]}) "
                f"does not match the original number of feature names ({len(feature_names)}).\n"
                f"Missing features: {missing_features}\n"
                f"Extra features: {extra_features}"
            )

        # Define best parameters for RandomForestRegressor
        best_params_rf = {
            'n_estimators': 300,
            'max_depth': 30,
            'min_samples_split': 10,
            'min_samples_leaf': 4
        }

        # Initialize and train the Random Forest model with the best parameters
        rf_model = RandomForestRegressor(
            n_estimators=best_params_rf['n_estimators'],
            max_depth=best_params_rf['max_depth'],
            min_samples_split=best_params_rf['min_samples_split'],
            min_samples_leaf=best_params_rf['min_samples_leaf'],
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        ## Calcular la importància de les característiques
        importances = rf_model.feature_importances_
        

        # Crear una sèrie amb la importància de les característiques, ordenades descendentment
        feature_importance = pd.Series(importances, index=feature_names[:X_train.shape[1]]).sort_values(ascending=False)
        

        # Seleccionar les 65 característiques més importants per a la predicció
        significant_variables = feature_importance.head(65).index.tolist()

        # Mostrar les caixes d'entrada per a les variables significatives en una graella
        st.header("Valors obtinguts per a la predicció:")
        input_values = {}
        num_columns = 5
        num_rows = (len(significant_variables) + num_columns - 1) // num_columns  # Calcular el nombre de files necessàries

        for row in range(num_rows):
            cols = st.columns(num_columns)
            for i, col in enumerate(cols):
                index = row * num_columns + i
                if index < len(significant_variables):
                    feature = significant_variables[index]
                    input_values[feature] = col.number_input(feature)

        # Preparar les dades d'entrada com a DataFrame per a la predicció
        input_data = pd.DataFrame([input_values])

        # Aplicar el preprocessor a les dades d'entrada
        input_data_preprocessed = pipeline.transform(input_data)

        # Realitzar la predicció utilitzant el model entrenat
        prediction = rf_model.predict(input_data_preprocessed)

        # Mostrar el resultat de la predicció
        st.subheader(f"Predicció per '{selected_y}':")
        st.write(prediction[0])
       
    

    


    #######    
