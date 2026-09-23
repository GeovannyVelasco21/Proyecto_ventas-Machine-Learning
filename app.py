# app.py — Interfaz Streamlit: predicción de la venta del próximo mes por asesor
# Autor: Geovanny Andrés Velasco Ospina — desarrollo completo
import json                                          # Leer archivos JSON
from pathlib import Path                             # Rutas relativas a este archivo

import joblib                                        # Cargar el modelo entrenado
import pandas as pd                                  # Tablas
import streamlit as st                               # Interfaz web

import utils                                         # Misma ingeniería de características del notebook (R10)

CARPETA = Path(__file__).parent                      # Carpeta donde está app.py (funciona local y en la nube)

# ---- Resultados de la evaluación (Notebook 02, Fase 9) ----
MODELOS_CV = pd.DataFrame({                          # Validación cruzada temporal (2024–2025): base de la selección
    'Modelo': ['Línea base (mes anterior)', 'Ridge', 'Árbol de regresión', 'Random Forest ✅'],
    'MAE (pesos)': [22_536_814, 20_359_150, 18_829_673, 17_213_250],
    'R²': [0.45, 0.55, 0.63, 0.68],
    'Brecha train–validación': ['—', '7,0 %', '18,5 %', '20,0 %']})
MODELOS_2026 = pd.DataFrame({                        # Examen final (ene–ago 2026, datos nunca vistos)
    'Modelo': ['Línea base (mes anterior)', 'Ridge', 'Árbol de regresión', 'Random Forest ✅'],
    'MAE (pesos)': [26_309_923, 20_541_269, 16_805_565, 15_668_156],
    'R²': [-0.08, 0.42, 0.61, 0.65],
    'Mejora vs. línea base': ['—', '21,9 %', '36,1 %', '40,4 %']})

MAE_PRUEBA = 15_668_156                              # Error promedio del Random Forest en la prueba 2026 (Fase 9)
MAE_NOV_DIC_CV = 31_304_241                          # Error en nov–dic 2025 (validación cruzada, fold 5)

# ---- Rangos observados en los datos (Notebook 01): la app no acepta valores fuera de ellos ----
RANGOS = {'edad_asesor': (14, 57), 'total_dias_absentismo': (0, 31),
          'metros_cuadrados_tienda': (50.0, 329.0), 'meses_antiguedad': (0, 28)}

# ---- Niveles de confiabilidad según el error relativo esperado (criterio propio del proyecto) ----
NIVELES = [(0.25, 'success', '🟢 Confiabilidad alta'),
           (0.50, 'warning', '🟡 Confiabilidad media'),
           (float('inf'), 'error', '🔴 Confiabilidad baja')]


@st.cache_resource                                   # Carga el modelo UNA sola vez, no en cada clic
def cargar_modelo():
    return joblib.load(CARPETA / 'modelo_final.joblib')


@st.cache_data                                       # Lee un JSON una sola vez
def cargar_json(nombre):
    with open(CARPETA / nombre, encoding='utf-8') as f:
        return json.load(f)


def pesos(valor):
    """Formato colombiano: $ 37.942.924"""
    return '$ ' + f'{valor:,.0f}'.replace(',', '.')


st.set_page_config(page_title='Predicción de venta por asesor', page_icon='📈', layout='centered')
modelo = cargar_modelo()
variables = cargar_json('variables_seleccionadas.json')
PREDICTORES = variables['numericas'] + variables['binarias'] + variables['categoricas']

# Categorías exactas que vio el modelo al entrenar (se leen del propio modelo)
codificador = modelo.regressor_.named_steps['prep'].named_transformers_['cat']
CATEGORIAS = {col: [str(c) for c in cats]
              for col, cats in zip(variables['categoricas'], codificador.categories_)}

st.title('📈 Predicción de la venta del próximo mes')
st.caption('Random Forest entrenado con datos asesor-mes (abril 2024 – agosto 2026).')

tab_pred, tab_modelos, tab_proceso = st.tabs(['🔮 Predicción', '📊 Modelos evaluados', '🧭 ¿Cómo llegamos aquí?'])

# =========================== PESTAÑA 1: PREDICCIÓN ===========================
with tab_pred:
    with st.form('formulario'):
        st.subheader('1. Mes a predecir y tienda')
        c1, c2 = st.columns(2)
        mes = c1.selectbox('Mes a predecir', utils.MESES, index=utils.MESES.index('septiembre'))
        marca = c2.selectbox('Marca', CATEGORIAS['marca'])
        metros = st.number_input('Metros cuadrados de la tienda', min_value=RANGOS['metros_cuadrados_tienda'][0],
                                 max_value=RANGOS['metros_cuadrados_tienda'][1], value=140.0, step=1.0)

        st.subheader('2. Asesor')
        c1, c2 = st.columns(2)
        edad = c1.number_input('Edad (años)', min_value=RANGOS['edad_asesor'][0],
                               max_value=RANGOS['edad_asesor'][1], value=29, step=1)
        nacionalidad = c2.selectbox('Nacionalidad', CATEGORIAS['nacionalidad_asesor'])
        c1, c2 = st.columns(2)
        antiguedad = c1.number_input('Meses de antigüedad', min_value=RANGOS['meses_antiguedad'][0],
                                     max_value=RANGOS['meses_antiguedad'][1], value=6, step=1,
                                     help='Meses que el asesor ha trabajado antes del mes a predecir.')
        absentismo = c2.number_input('Días de absentismo del mes', min_value=RANGOS['total_dias_absentismo'][0],
                                     max_value=RANGOS['total_dias_absentismo'][1], value=0, step=1)

        st.subheader('3. Venta de los meses anteriores (pesos)')
        st.caption('Escriba 0 si el asesor no trabajó ese mes (sin historia).')
        c1, c2, c3 = st.columns(3)
        v1 = c1.number_input('Mes anterior', min_value=0, value=0, step=1_000_000)
        v2 = c2.number_input('Hace 2 meses', min_value=0, value=0, step=1_000_000)
        v3 = c3.number_input('Hace 3 meses', min_value=0, value=0, step=1_000_000)

        enviado = st.form_submit_button('Predecir', type='primary', width='stretch')

    if enviado:
        entrada = pd.DataFrame([{                    # Un registro con los datos digitados
            'edad_asesor': float(edad), 'total_dias_absentismo': float(absentismo),
            'metros_cuadrados_tienda': float(metros), 'meses_antiguedad': float(antiguedad),
            'venta_t_1': float(v1), 'venta_t_2': float(v2), 'venta_t_3': float(v3),
            'nacionalidad_asesor': nacionalidad, 'marca': marca, 'mes_venta': mes}])
        X = utils.preparar_features(entrada)[PREDICTORES]   # Misma transformación que en el entrenamiento
        prediccion = float(modelo.predict(X)[0])

        st.metric('Venta estimada', pesos(prediccion))

        # ---- Alerta: % de error esperado = MAE de la prueba 2026 / venta estimada ----
        relativo = MAE_PRUEBA / max(prediccion, 1)   # Qué tanto representa el error promedio frente a esta predicción
        _, tipo_alerta, etiqueta = next(n for n in NIVELES if relativo <= n[0])
        getattr(st, tipo_alerta)(
            f'**{etiqueta}.** El error promedio del modelo en 2026 fue de **{pesos(MAE_PRUEBA)}**, '
            f'que equivale a ≈ **{relativo:.0%}** de esta estimación. Rango orientativo: '
            f'**{pesos(max(prediccion - MAE_PRUEBA, 0))}** a **{pesos(prediccion + MAE_PRUEBA)}**.')
        if mes == 'diciembre':
            st.warning(f'📅 Diciembre es el mes de mayor venta y la prueba 2026 no lo incluye. En la validación '
                       f'cruzada (noviembre–diciembre 2025) el error promedio fue {pesos(MAE_NOV_DIC_CV)}: '
                       f'tome esta estimación con mayor cautela.')
        if v1 == 0:
            st.info('La venta del mes anterior es 0: el modelo trata al asesor como **sin historia** '
                    'y estima con su perfil, la tienda y el mes.')
        st.caption('El rango es una referencia práctica basada en el error promedio observado, no un intervalo '
                   'de confianza estadístico. Niveles: 🟢 error esperado ≤ 25 % · 🟡 25–50 % · 🔴 > 50 % '
                   '(criterio propio del proyecto).')

# ======================== PESTAÑA 2: MODELOS EVALUADOS ========================
with tab_modelos:
    st.subheader('Tres modelos comparados contra una línea base')
    st.markdown('La **línea base** es la regla más simple: *"el asesor vende lo mismo que el mes anterior"*. '
                'Un modelo solo se justifica si la supera. **MAE** = error promedio en pesos (menor es mejor); '
                '**R²** = proporción de la variación de la venta que se explica (1 es perfecto).')
    formato = {'MAE (pesos)': lambda v: pesos(v), 'R²': '{:.2f}'}
    st.markdown('**1. Validación cruzada temporal (2024–2025)** — con esto se eligió el modelo:')
    st.dataframe(MODELOS_CV.style.format(formato), hide_index=True, width='stretch')
    st.markdown('**2. Examen final (enero–agosto 2026, datos nunca vistos)** — confirma la elección:')
    st.dataframe(MODELOS_2026.style.format(formato), hide_index=True, width='stretch')
    st.success('**Random Forest** fue el mejor en ambas evaluaciones: reduce el error un 40 % frente a la '
               'línea base y explica el 65 % de la variación de la venta en 2026.')
    imagen = CARPETA / 'assets' / 'importancia_variables.png'
    if imagen.exists():
        st.image(str(imagen), caption='Qué pesa más en la predicción: la venta de los 3 meses anteriores '
                                      '(74 %); el mes, el perfil del asesor y la tienda afinan la estimación.')

# ====================== PESTAÑA 3: ¿CÓMO LLEGAMOS AQUÍ? ======================
with tab_proceso:
    st.subheader('Metodología CRISP-DM, en 6 pasos')
    st.markdown(
        '1. **Entender los datos** — 17.603 registros asesor-mes (abril 2024 – agosto 2026), 2.237 asesores y '
        '151 tiendas. Se verificó que no hubiera vacíos, duplicados ni meses faltantes.\n'
        '2. **Preparar** — variables nuevas: *sin historia* (distingue un asesor nuevo de uno que vendió poco), '
        '*antigüedad* y la venta anterior en escala logarítmica (para que los pocos asesores de ventas muy altas '
        'no dominen el modelo).\n'
        '3. **Elegir las variables** — 4 métodos (correlación, índice de ganancia, árbol de decisión y regresión '
        'logística) votaron; quedaron 11 de 18 variables.\n'
        '4. **Evaluar sin hacer trampa** — se entrenó con 2024–2025 y se evaluó con 2026, respetando el orden '
        'del tiempo: el modelo nunca vio el futuro al aprender.\n'
        '5. **Comparar modelos** — Ridge, árbol y Random Forest, cuidando que no memorizaran los datos '
        '(sobreajuste). Ganó Random Forest.\n'
        '6. **Desplegar** — el modelo final se reentrenó con todos los datos y se publicó en esta app.')
    st.subheader('Limitaciones')
    st.markdown(
        '- Predice **un mes adelante**; para meses posteriores se necesita la venta real del mes previo.\n'
        '- En promedio **subestima** a los asesores de venta alta.\n'
        '- Los errores relativos son grandes cuando la venta real es muy baja.\n'
        '- Muestra asociaciones, no causas.')

st.divider()
st.caption('Autor: Geovanny Andrés Velasco Ospina · Maestría en Ciencia de Datos · '
           'Proyecto académico con metodología CRISP-DM')
