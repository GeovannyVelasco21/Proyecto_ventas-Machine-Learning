# app.py — Interfaz Streamlit: predicción de la venta del próximo mes por asesor
# Autor: Geovanny Andrés Velasco Ospina — desarrollo completo
import json                                          # Leer la lista de variables
from pathlib import Path                             # Rutas relativas a este archivo

import joblib                                        # Cargar el modelo entrenado
import pandas as pd                                  # Tablas
import streamlit as st                               # Interfaz web

import utils                                         # Misma ingeniería de características del notebook (R10)

CARPETA = Path(__file__).parent                      # Carpeta donde está app.py (funciona local y en la nube)

# ---- Métricas del examen final (prueba ene–ago 2026, Notebook 02, Fase 9) ----
MAE_PRUEBA = 15_668_156                              # Error absoluto medio del Random Forest en 2026
METRICAS = pd.DataFrame({
    'Modelo': ['Línea base (mes anterior)', 'Ridge', 'Árbol de regresión', 'Random Forest (seleccionado)'],
    'MAE (pesos)': [26_309_923, 20_541_269, 16_805_565, 15_668_156],
    'R²': [-0.08, 0.42, 0.61, 0.65]})

# ---- Rangos observados en los datos (Notebook 01): la app no acepta valores fuera de ellos ----
RANGOS = {'edad_asesor': (14, 57), 'total_dias_absentismo': (0, 31),
          'metros_cuadrados_tienda': (50.0, 329.0), 'meses_antiguedad': (0, 28)}


@st.cache_resource                                   # Carga el modelo UNA sola vez, no en cada clic
def cargar_modelo():
    return joblib.load(CARPETA / 'modelo_final.joblib')


@st.cache_data                                       # Carga la lista de variables una sola vez
def cargar_variables():
    with open(CARPETA / 'variables_seleccionadas.json', encoding='utf-8') as f:
        return json.load(f)


def pesos(valor):
    """Formato colombiano: $ 37.942.924"""
    return '$ ' + f'{valor:,.0f}'.replace(',', '.')


st.set_page_config(page_title='Predicción de venta por asesor', page_icon='📈', layout='centered')
modelo = cargar_modelo()
variables = cargar_variables()
PREDICTORES = variables['numericas'] + variables['binarias'] + variables['categoricas']

# Categorías exactas que vio el modelo al entrenar (se leen del propio modelo)
codificador = modelo.regressor_.named_steps['prep'].named_transformers_['cat']
CATEGORIAS = {col: [str(c) for c in cats]
              for col, cats in zip(variables['categoricas'], codificador.categories_)}

st.title('📈 Predicción de la venta del próximo mes')
st.caption('Random Forest entrenado con datos asesor-mes (abril 2024 – agosto 2026). '
           'Ingrese los datos del asesor y su venta de los 3 meses anteriores.')

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
    entrada = pd.DataFrame([{                        # Un registro con los datos digitados
        'edad_asesor': float(edad), 'total_dias_absentismo': float(absentismo),
        'metros_cuadrados_tienda': float(metros), 'meses_antiguedad': float(antiguedad),
        'venta_t_1': float(v1), 'venta_t_2': float(v2), 'venta_t_3': float(v3),
        'nacionalidad_asesor': nacionalidad, 'marca': marca, 'mes_venta': mes}])
    X = utils.preparar_features(entrada)[PREDICTORES]   # Misma transformación que en el entrenamiento
    prediccion = float(modelo.predict(X)[0])

    st.success(f'**Venta estimada:** {pesos(prediccion)}')
    minimo = max(prediccion - MAE_PRUEBA, 0)
    st.write(f'Rango orientativo (± error promedio de la prueba): '
             f'**{pesos(minimo)}** a **{pesos(prediccion + MAE_PRUEBA)}**')
    st.caption('El rango se basa en el error absoluto medio (MAE) del modelo en 2026. '
               'Es una referencia práctica, no un intervalo de confianza estadístico.')
    if v1 == 0:
        st.info('La venta del mes anterior es 0: el modelo trata al asesor como **sin historia** '
                'y estima con su perfil, la tienda y el mes.')

with st.expander('¿Cómo funciona?'):
    st.markdown(
        '- **Modelo:** Random Forest (200 árboles), elegido entre 3 modelos con validación cruzada temporal.\n'
        '- **Datos:** 17.603 registros asesor-mes, abril 2024 – agosto 2026.\n'
        '- **Qué pesa más:** la venta de los 3 meses anteriores (74 % de la importancia); '
        'el mes, el perfil del asesor y la tienda afinan la estimación.\n'
        '- **Limitaciones:** predice un mes adelante; en promedio subestima a los asesores de venta alta; '
        'los errores relativos son grandes cuando la venta real es muy baja.')
    st.markdown('**Desempeño en datos nunca vistos (enero–agosto 2026):**')
    st.dataframe(METRICAS.style.format({'MAE (pesos)': lambda v: pesos(v), 'R²': '{:.2f}'}),
                 hide_index=True, width='stretch')
    imagen = CARPETA / 'assets' / 'importancia_variables.png'
    if imagen.exists():
        st.image(str(imagen), caption='Importancia de variables del Random Forest')

st.caption('Autor: Geovanny Andrés Velasco Ospina · Maestría en Ciencia de Datos')
