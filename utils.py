# utils.py — Ingeniería de características (la usan el notebook y la app)
import numpy as np                                   # Para log1p
import pandas as pd                                  # Para tablas y tipos

MESES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
         'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']   # Orden del calendario
CATEGORICAS = ['genero_asesor', 'tipo_vinculacion', 'nacionalidad_asesor',
               'tipo_ubicacion_tienda', 'marca', 'zona_comercial', 'mes_venta']   # Predictores categóricos
REZAGOS = ['venta_t_1', 'venta_t_2', 'venta_t_3']    # Ventas de los 3 meses anteriores


def agregar_antiguedad(df):
    """Meses que lleva el asesor en la base (solo notebook; en la app se digita)."""
    out = df.sort_values(['id_asesor', 't']).copy()  # Orden asesor-tiempo: requisito para contar bien
    out['meses_antiguedad'] = (out.groupby('id_asesor', observed=True)  # Por cada asesor...
                                  .cumcount()        # ...cuenta sus registros previos: 0, 1, 2... (solo pasado, sin fuga)
                                  .astype('float64'))
    return out


def preparar_features(df):
    """Crea sin_hist y log de rezagos, y estandariza categorías. Misma lógica para notebook y app."""
    out = df.copy()                                  # No modifica la tabla original
    for k, col in enumerate(REZAGOS, start=1):       # Recorre venta_t_1, t_2, t_3
        venta = out[col].astype('float64')           # Asegura tipo numérico
        out[f'sin_hist_t_{k}'] = (venta == 0).astype('float64')   # 1 = sin historia ese mes; 0 = sí tiene
        out[f'log_{col}'] = np.log1p(venta)          # log(1 + venta): reduce asimetría; el 0 queda en 0
    out['marca'] = (pd.to_numeric(out['marca'].astype(str))   # '2' o 2 → 2...
                      .astype(int).astype(str))      # ...→ '2' (mismo formato en notebook y app)
    out['mes_venta'] = out['mes_venta'].astype(str).str.lower().str.strip()   # 'Diciembre ' → 'diciembre'
    for c in CATEGORICAS:                            # Estandariza cada categórica
        if c not in out.columns:                     # La app solo envía las categóricas seleccionadas
            continue
        if c == 'mes_venta':
            out[c] = pd.Categorical(out[c], categories=MESES, ordered=True)   # Mes en orden de calendario
        else:
            out[c] = out[c].astype(str).astype('category')                   # Texto → category
    return out
