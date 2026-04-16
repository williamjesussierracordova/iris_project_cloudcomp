import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd
import math
from datetime import datetime, timezone

import psycopg2
from psycopg2 import sql
# Fetch variables
USER = "postgres.wowffrsbanthdyhquzjg" #os.getenv("user")
PASSWORD = "iris_project12"# os.getenv("password")
HOST = "aws-1-us-west-2.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")
TABLE_NAME = "iris"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")


def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )


def ensure_predictions_table():
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        id BIGSERIAL PRIMARY KEY,
                        sepal_length DOUBLE PRECISION NOT NULL,
                        sepal_width DOUBLE PRECISION NOT NULL,
                        petal_length DOUBLE PRECISION NOT NULL,
                        petal_width DOUBLE PRECISION NOT NULL,
                        predicted_species TEXT NOT NULL,
                        confidence DOUBLE PRECISION,
                        prob_setosa DOUBLE PRECISION,
                        prob_versicolor DOUBLE PRECISION,
                        prob_virginica DOUBLE PRECISION,
                        fecha_prediccion TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
            connection.commit()
        return True, None
    except Exception as e:
        return False, str(e)


def get_table_columns(table_name=TABLE_NAME):
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (table_name,)
            )
            return {row[0] for row in cursor.fetchall()}


def save_prediction(prediction_data):
    try:
        table_columns = get_table_columns()
        payload = {
            "sepal_length": float(prediction_data["sepal_length"]),
            "sepal_width": float(prediction_data["sepal_width"]),
            "petal_length": float(prediction_data["petal_length"]),
            "petal_width": float(prediction_data["petal_width"]),
            "predicted_species": prediction_data["predicted_species"],
            "species": prediction_data["predicted_species"],
            "confidence": float(prediction_data["confidence"]),
            "fecha_prediccion": prediction_data.get("fecha_prediccion", datetime.now(timezone.utc)),
            "prob_setosa": prediction_data.get("prob_setosa"),
            "prob_versicolor": prediction_data.get("prob_versicolor"),
            "prob_virginica": prediction_data.get("prob_virginica"),
        }

        insert_columns = [col for col in payload.keys() if col in table_columns]
        if not insert_columns:
            return False, "No hay columnas compatibles para insertar en la tabla."

        values = [payload[col] for col in insert_columns]
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(TABLE_NAME),
            sql.SQL(", ").join([sql.Identifier(col) for col in insert_columns]),
            sql.SQL(", ").join([sql.Placeholder() for _ in insert_columns]),
        )

        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, values)
            connection.commit()

        st.cache_data.clear()
        return True, None
    except Exception as e:
        return False, str(e)


@st.cache_data(ttl=30, show_spinner=False)
def load_prediction_history():
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SELECT * FROM {} ORDER BY fecha_prediccion DESC").format(
                        sql.Identifier(TABLE_NAME)
                    )
                )
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

        df = pd.DataFrame(rows, columns=columns)
        if "fecha_prediccion" in df.columns:
            df["fecha_prediccion"] = pd.to_datetime(df["fecha_prediccion"], utc=True, errors="coerce")
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)



# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'models/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()
db_ready, db_error = ensure_predictions_table()

if not db_ready:
    st.error(f"No se pudo preparar la tabla de histórico en Supabase: {db_error}")

if model is not None:
    # Inputs
    st.header("Ingresa las características de la flor:")
    
    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Botón de predicción
    if st.button("Predecir Especie"):
        # Preparar datos
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Estandarizar
        features_scaled = scaler.transform(features)
        
        # Predecir
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Mostrar resultado
        target_names = model_info['target_names']
        predicted_species = target_names[prediction]
        
        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{max(probabilities):.1%}**")

        if db_ready:
            prediction_row = {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "predicted_species": predicted_species,
                "confidence": float(max(probabilities)),
                "fecha_prediccion": datetime.now(timezone.utc),
            }

            for species, prob in zip(target_names, probabilities):
                probability_column = f"prob_{species.lower().replace(' ', '_')}"
                prediction_row[probability_column] = float(prob)

            ok, error_msg = save_prediction(prediction_row)
            if ok:
                st.caption("Predicción guardada en Supabase.")
            else:
                st.warning(f"No se pudo guardar la predicción en Supabase: {error_msg}")
        
        # Mostrar todas las probabilidades
        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")


st.markdown("---")
st.header("Histórico de predicciones")

history_df, history_error = load_prediction_history()

if history_error:
    st.error(f"No se pudo cargar el histórico: {history_error}")
elif history_df.empty:
    st.info("Aún no hay predicciones registradas.")
else:
    filtered_df = history_df.copy()
    species_column = "predicted_species" if "predicted_species" in filtered_df.columns else "species" if "species" in filtered_df.columns else None

    filter_col_1, filter_col_2, filter_col_3 = st.columns(3)

    with filter_col_1:
        species_filter = "Todas"
        if species_column is not None:
            species_options = ["Todas"] + sorted(filtered_df[species_column].dropna().astype(str).unique().tolist())
            species_filter = st.selectbox("Filtrar por especie", species_options)

    with filter_col_2:
        start_date = None
        end_date = None
        if "fecha_prediccion" in filtered_df.columns and filtered_df["fecha_prediccion"].notna().any():
            min_date = filtered_df["fecha_prediccion"].min().date()
            max_date = filtered_df["fecha_prediccion"].max().date()
            date_range = st.date_input("Rango de fecha", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range

    with filter_col_3:
        search_text = st.text_input("Buscar texto", "").strip().lower()

    if species_column is not None and species_filter != "Todas":
        filtered_df = filtered_df[filtered_df[species_column].astype(str) == species_filter]

    if start_date is not None and end_date is not None and "fecha_prediccion" in filtered_df.columns:
        mask_dates = filtered_df["fecha_prediccion"].dt.date.between(start_date, end_date)
        filtered_df = filtered_df[mask_dates]

    if search_text:
        text_mask = filtered_df.astype(str).apply(lambda col: col.str.lower().str.contains(search_text, na=False))
        filtered_df = filtered_df[text_mask.any(axis=1)]

    sort_default_idx = 0
    if "fecha_prediccion" in filtered_df.columns:
        sort_default_idx = filtered_df.columns.get_loc("fecha_prediccion")

    sort_col_1, sort_col_2 = st.columns(2)
    with sort_col_1:
        sort_by = st.selectbox("Ordenar por columna", options=filtered_df.columns.tolist(), index=sort_default_idx)
    with sort_col_2:
        ascending = st.checkbox("Orden ascendente", value=False)

    filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    page_size = 10
    total_rows = len(filtered_df)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page_number = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    page_df = filtered_df.iloc[start_idx:end_idx].copy()

    if "fecha_prediccion" in page_df.columns:
        page_df["fecha_prediccion"] = page_df["fecha_prediccion"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    st.caption(f"Mostrando {len(page_df)} de {total_rows} registros (página {page_number} de {total_pages}).")
    st.dataframe(page_df, use_container_width=True, hide_index=True)


st.markdown("---")
st.header("Métricas")

if history_error:
    st.info("Métricas no disponibles por error de conexión al histórico.")
elif history_df.empty:
    st.info("No hay datos para calcular métricas.")
else:
    metrics_df = history_df.copy()
    species_column = "predicted_species" if "predicted_species" in metrics_df.columns else "species" if "species" in metrics_df.columns else None

    total_predictions = len(metrics_df)
    unique_species = metrics_df[species_column].nunique() if species_column else 0
    avg_confidence = pd.to_numeric(metrics_df["confidence"], errors="coerce").mean() if "confidence" in metrics_df.columns else np.nan
    last_prediction = metrics_df["fecha_prediccion"].max() if "fecha_prediccion" in metrics_df.columns else None

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Total de predicciones", f"{total_predictions}")
    metric_col_2.metric("Especies predichas", f"{unique_species}")
    metric_col_3.metric("Confianza promedio", f"{avg_confidence:.2%}" if pd.notna(avg_confidence) else "N/A")
    metric_col_4.metric(
        "Última predicción",
        last_prediction.strftime("%Y-%m-%d %H:%M") if pd.notna(last_prediction) else "N/A"
    )

    if species_column is not None:
        st.subheader("Distribución de predicciones por especie")
        species_counts = metrics_df.groupby(species_column).size().sort_values(ascending=False)
        st.bar_chart(species_counts)

    if "fecha_prediccion" in metrics_df.columns:
        valid_dates_df = metrics_df[metrics_df["fecha_prediccion"].notna()].copy()
        if not valid_dates_df.empty:
            valid_dates_df["fecha"] = valid_dates_df["fecha_prediccion"].dt.date
            predictions_per_day = valid_dates_df.groupby("fecha").size().rename("predicciones")
            st.subheader("Predicciones por día")
            st.line_chart(predictions_per_day)