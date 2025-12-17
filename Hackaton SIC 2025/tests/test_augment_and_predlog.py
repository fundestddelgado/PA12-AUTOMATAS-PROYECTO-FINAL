import os
import tempfile
import pandas as pd
import numpy as np
from src.data.data_loader import augment_dataset


def build_base_df(n=1000, frac_failure=0.05, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'air_temp_k': rng.normal(300, 2, n),
        'process_temp_k': rng.normal(310, 2, n),
        'rot_speed_rpm': rng.normal(1500, 100, n),
        'torque_nm': rng.normal(40, 2, n),
        'tool_wear_min': np.abs(rng.normal(20, 5, n)),
        'type': ['M'] * n,
        'machine_failure': rng.binomial(1, frac_failure, n)
    })
    return df


def test_augment_balance_mode_matches_target_ratio(tmp_path=None):
    df = build_base_df(n=1000, frac_failure=0.05)
    # Generar 1000 filas sintéticas y apuntar a una razón total de 0.1
    out = augment_dataset(df, n=1000, seed=1, mode='balance', target_failure_ratio=0.1)
    combined = pd.concat([df, out], ignore_index=True)
    frac = combined['machine_failure'].mean()
    # Debe estar cercano a 0.1 (permitir cierta tolerancia de redondeo)
    assert abs(frac - 0.1) < 0.02


def test_augment_targeted_feature_injection():
    df = build_base_df(n=1000, frac_failure=0.05)
    # Inyectar un valor específico (air_temp_k = 500) en 10% de las filas
    out = augment_dataset(df, n=500, seed=2, mode='targeted', targeted_feature='air_temp_k', targeted_value=500.0, targeted_frac=0.1)
    count_targeted = (out['air_temp_k'] == 500.0).sum()
    assert abs(count_targeted - int(round(500 * 0.1))) <= 2


def test_load_dataset_includes_additional(tmp_path):
    # crear un CSV base temporal y un CSV adicional; asegurar que load_dataset pueda combinarlos
    from src.data.data_loader import load_dataset
    base = tmp_path / 'ai4i2020.csv'
    df_base = build_base_df(n=50, frac_failure=0.05)
    df_base.to_csv(base, index=False)
    add_dir = tmp_path / 'additional'
    add_dir.mkdir()
    df_add = build_base_df(n=10, frac_failure=0.2)
    df_add.to_csv(add_dir / 'extra.csv', index=False)
    df_no = load_dataset(path=str(base), include_additional=False)
    df_yes = load_dataset(path=str(base), include_additional=True, additional_dir=str(add_dir))
    assert len(df_yes) == len(df_no) + 10
