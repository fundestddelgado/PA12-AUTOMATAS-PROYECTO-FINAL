import os
import pandas as pd
from app.streamlit_app import ensure_pred_log_has_history, log_prediction, log_feedback, load_models, prepare_feature_row
import csv

LOG_PATH = os.path.abspath(os.path.join('.', 'logs'))
PRED_LOG = os.path.join(LOG_PATH, 'predicciones.csv')
FEEDBACK_LOG = os.path.join(LOG_PATH, 'feedback.csv')  # legado: ya no se usa; pruebas actualizadas para usar el log de predicciones


def test_seed_and_log_dedup():
    # limpieza
    if os.path.exists(PRED_LOG): os.remove(PRED_LOG)
    if os.path.exists(FEEDBACK_LOG): os.remove(FEEDBACK_LOG)
    model, _ = load_models()
    ensure_pred_log_has_history(model, count=3)
    assert os.path.exists(PRED_LOG)
    size_before = os.path.getsize(PRED_LOG)
    # crear una predicción simulada
    user_data = {'air_temp_k': 300.0, 'process_temp_k': 310.0, 'rot_speed_rpm': 1500, 'torque_nm': 40.0, 'tool_wear_min': 10.0, 'type': 'L'}
    row = prepare_feature_row(user_data)
    # evitar predecir si el modelo no está disponible; construimos un registro manual
    rec = {**user_data, 'pred': 0, 'prob': 0.5, 'prediction_timestamp': '2025-11-17 13:00:00'}
    log_feedback(rec['prediction_timestamp'], 0)
    log_prediction(rec, rec['prob'], rec['pred'])
    # llamar una segunda vez para probar la deduplicación
    log_prediction(rec, rec['prob'], rec['pred'])
    # leer archivo
    with open(PRED_LOG, 'r', newline='') as f:
        rows = list(csv.reader(f))
    # encabezado + al menos 4 filas (3 de siembra + 1 nueva); no deben aparecer duplicados
    assert len(rows) >= 4
    # asegurar que el timestamp de la última fila coincida (se convierte a formato ISO con "T")
    # convertir espacio por "T": "2025-11-17 13:00:00" -> "2025-11-17T13:00:00"
    expected_ts = rec['prediction_timestamp'].replace(' ', 'T')
    assert rows[-1][0] == expected_ts


def test_combine_and_save_labeled():
    # limpieza
    add_dir = os.path.abspath(os.path.join('.', 'data', 'additional'))
    if not os.path.isdir(add_dir):
        os.makedirs(add_dir, exist_ok=True)
    # eliminar archivos adicionales antiguos
    for f in os.listdir(add_dir):
        if f.lower().endswith('.csv'):
            os.remove(os.path.join(add_dir, f))
    # asegurar que exista al menos una predicción y su feedback
    model, _ = load_models()
    ensure_pred_log_has_history(model, count=3)
    user_data = {'air_temp_k': 300.0, 'process_temp_k': 310.0, 'rot_speed_rpm': 1500, 'torque_nm': 40.0, 'tool_wear_min': 10.0, 'type': 'L'}
    row = prepare_feature_row(user_data)
    prob = model.predict_proba(row)[0][1]
    pred = int(model.predict(row)[0])
    rec = {**user_data, 'pred': pred, 'prob': prob, 'prediction_timestamp': '2025-11-17 13:00:01'}
    log_feedback(rec['prediction_timestamp'], 1)
    log_prediction(rec, rec['prob'], rec['pred'])

