import streamlit as st
import pandas as pd
import os

# Define la ruta base del proyecto de forma robusta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Nombres de Columnas de Datos (omisión por brevedad)
    STATION_NAME_COL = 'nom_est'
    PRECIPITATION_COL = 'precipitation'
    LATITUDE_COL = 'latitud_geo'
    LONGITUDE_COL = 'longitud_geo'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    DATE_COL = 'fecha_mes_año'
    ENSO_ONI_COL = 'anomalia_oni'
    ORIGIN_COL = 'origen'
    ALTITUDE_COL = 'alt_est'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    CELL_COL = 'celda_xy'
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    # Rutas de Archivos (usando la ruta absoluta)
    # FIX FINAL: Ruta sin espacios para máxima compatibilidad
    LOGO_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde_Logo.jpg") 
    LOGO_DROP_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde_Logo.jpg") 
    GIF_PATH = os.path.join(BASE_DIR, "data", "PPAM.gif")
    
    # Mensajes de la UI (omisión por brevedad)
    APP_TITLE = "Sistema de información de las lluvias y el Clima en el norte de la región Andina"
    WELCOME_TEXT = "..."

    @staticmethod
    def initialize_session_state():
        # ... (omisión por brevedad)
        pass
