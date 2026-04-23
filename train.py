def entrenar():
    import pandas as pd
    import joblib

    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    MODELO_PATH = "modelo_ciclismo.pkl"

    # Cargar datos
    data = pd.read_csv("dataset_ciclismo_fatiga.csv")

    # Variables
    X = data[["frecuencia_cardiaca", "potencia", "cadencia", "tiempo",
              "temperatura", "pendiente", "velocidad"]]
    y = data["fatiga"]

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline
    modelo = Pipeline([
        ("modelo", DecisionTreeRegressor(max_depth=5, random_state=42))
    ])

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Evaluación
    y_pred = modelo.predict(X_test)

    print("Evaluación del modelo:")
    print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("R2:", round(r2_score(y_test, y_pred), 4))

    # Guardar modelo
    joblib.dump(modelo, MODELO_PATH)
    print("\nModelo entrenado y guardado correctamente.")
    
    return (
    round(mean_squared_error(y_test, y_pred), 2),
    round(mean_absolute_error(y_test, y_pred), 2),
    round(r2_score(y_test, y_pred), 4)
    )
    

