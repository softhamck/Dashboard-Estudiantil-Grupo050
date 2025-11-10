import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# === Configuración general ===
st.set_page_config(page_title="Dashboard Estudiantil – Grupo 050", layout="wide")

# --- Tema oscuro global para matplotlib ---
plt.style.use("dark_background")

# === Utilidades de carga ===
def _safe_read_excel(file_or_path):
    try:
        return pd.read_excel(file_or_path, engine="openpyxl")
    except Exception as e:
        st.error(f"No se pudo leer el archivo de Excel: {e}")
        raise

@st.cache_data
def load_data(file_or_path) -> pd.DataFrame:
    df = _safe_read_excel(file_or_path)
    return df.copy()

# === Normalización / limpieza ===
def _normalize_signs(txt: str) -> str:
    if txt is None or pd.isna(txt):
        return np.nan
    s = str(txt).strip().upper()
    s = (s.replace("＋", "+").replace("﹢", "+")
           .replace("−", "-").replace("–", "-").replace("—", "-").replace("﹣", "-"))
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_rh_value(raw) -> str:
    if raw is None or pd.isna(raw):
        return np.nan
    s = _normalize_signs(raw)
    s = re.sub(r"[.,;:_]", "", s).strip()
    s = re.sub(r"\bPOS(ITIVO)?\b", "+", s)
    s = re.sub(r"\bNEG(ATIVO)?\b", "-", s)
    s = re.sub(r"\b(RH|TIPO)\b", "", s).strip()
    s = s.replace(" ", "").replace("0", "O")
    if re.fullmatch(r"[+\-](A|B|AB|O)", s):
        s = s[1:] + s[0]
    m = re.fullmatch(r"(AB|A|B|O)([+\-]?)", s)
    if not m:
        return np.nan
    grupo, signo = m.groups()
    if signo == "":
        signo = "+"
    return f"{grupo}{signo}"

def normalize_rh_column(df: pd.DataFrame, col: str = "RH") -> pd.DataFrame:
    if col in df:
        df[col] = df[col].apply(_normalize_rh_value)
    return df

def tidy_text_series(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        out = s.astype(str).str.strip()
        out = out.str.replace(r"\s+", " ", regex=True)
        out = out.str.title()
        return out
    return s

def sanitize_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Nombre_Estudiante", "Apellido_Estudiante", "Barrio_Residencia", "Color_Cabello"]:
        if col in df:
            df[col] = tidy_text_series(df[col])
    return df

# === Enriquecimiento numérico ===
def compute_age(fecha_nac):
    if pd.isna(fecha_nac):
        return np.nan
    fn = pd.to_datetime(fecha_nac, errors="coerce")
    if pd.isna(fn):
        return np.nan
    hoy = pd.Timestamp.today().normalize()
    years = hoy.year - fn.year - ((hoy.month, hoy.day) < (fn.month, fn.day))
    return int(years)

def ensure_estatura_cm(df: pd.DataFrame) -> pd.DataFrame:
    if "Estatura" not in df:
        return df
    est = pd.to_numeric(df["Estatura"], errors="coerce")
    if est.mean(skipna=True) < 3.0:
        df["Estatura_cm"] = est * 100.0
        df["Estatura_m"] = est
    else:
        df["Estatura_cm"] = est
        df["Estatura_m"] = est / 100.0
    return df

def clasifica_imc(imc):
    if pd.isna(imc):
        return np.nan
    if imc < 18.5:
        return "Bajo peso"
    elif 18.5 <= imc < 25:
        return "Adecuado"
    elif 25 <= imc < 30:
        return "Sobrepeso"
    elif 30 <= imc < 35:
        return "Obesidad grado 1"
    elif 35 <= imc < 40:
        return "Obesidad grado 2"
    else:
        return "Obesidad grado 3"

# === Helper de filtros tolerantes ===
def optional_isin(series: pd.Series, selected):
    """Si selected está vacío/None -> no filtra. Si tiene valores -> aplica isin."""
    if selected is None or len(selected) == 0:
        return pd.Series(True, index=series.index)
    return series.isin(selected)

# === Carga automática del archivo local ===
data_source = Path(__file__).with_name("ListadoDeEstudiantesGrupo_050.xlsx")
if not data_source.exists():
    st.error("❌ No se encontró el archivo 'ListadoDeEstudiantesGrupo_050.xlsx' en la carpeta del proyecto.")
    st.stop()

df = load_data(data_source)
df = normalize_rh_column(df, col="RH")
df = sanitize_common_columns(df)

df["Nombre_Completo"] = (
    df.get("Nombre_Estudiante", "").astype(str).str.upper().str.strip()
    + " "
    + df.get("Apellido_Estudiante", "").astype(str).str.upper().str.strip()
)

df["Edad"] = df["Fecha_Nacimiento"].apply(compute_age) if "Fecha_Nacimiento" in df else np.nan
df = ensure_estatura_cm(df)
est_m = pd.to_numeric(df.get("Estatura_m", np.nan), errors="coerce").replace(0, np.nan)
peso = pd.to_numeric(df.get("Peso", np.nan), errors="coerce")
df["IMC"] = peso / (est_m ** 2)
df["Clasificación_IMC"] = df["IMC"].apply(clasifica_imc)

# === Filtros ===
with st.sidebar:
    st.markdown("### ⚙️ Filtros")

    integrantes_grupo = [
        "TODOS",
        "ANDREA MUÑOZ CANO",
        "CAMILO ANDRÉS FUENTES MORALES",
        "JULIANA MANCO HERRERA",
        "TOMAS MADRID GOMEZ",
    ]
    integrante_sel = st.selectbox("Integrante del grupo a exponer", integrantes_grupo, index=0)

    rh_vals = sorted(df["RH"].dropna().unique()) if "RH" in df else []
    rh_sel = st.multiselect("Tipo de Sangre (RH)", rh_vals, default=rh_vals)

    cab_vals = sorted(df["Color_Cabello"].dropna().unique()) if "Color_Cabello" in df else []
    cab_sel = st.multiselect("Color de Cabello", cab_vals, default=cab_vals)

    bar_vals = sorted(df["Barrio_Residencia"].dropna().unique()) if "Barrio_Residencia" in df else []
    bar_sel = st.multiselect("Barrio de Residencia", bar_vals, default=bar_vals)

    edades = df["Edad"].dropna()
    min_edad = int(edades.min()) if len(edades) else 0
    max_edad = int(edades.max()) if len(edades) else 0
    r_edad = st.slider("Rango de Edad", min_value=min_edad, max_value=max_edad, value=(min_edad, max_edad))

    ests = df["Estatura_cm"].dropna()
    min_est = int(ests.min()) if len(ests) else 0
    max_est = int(ests.max()) if len(ests) else 0
    r_est = st.slider("Rango de Estatura (cm)", min_value=min_est, max_value=max_est, value=(min_est, max_est))

# === Aplicar filtros===
mask = pd.Series(True, index=df.index)

if integrante_sel != "TODOS":
    mask &= (df["Nombre_Completo"] == integrante_sel.upper())

if "RH" in df:
    mask &= optional_isin(df["RH"], rh_sel)
if "Color_Cabello" in df:
    mask &= optional_isin(df["Color_Cabello"], cab_sel)
if "Barrio_Residencia" in df:
    mask &= optional_isin(df["Barrio_Residencia"], bar_sel)

mask &= df["Edad"].between(r_edad[0], r_edad[1])
mask &= df["Estatura_cm"].between(r_est[0], r_est[1])

dff = df.loc[mask].copy()

# === Título  ===
if integrante_sel == "TODOS":
    st.markdown("## 🎓 Dashboard Estudiantil – **Grupo 050**")
else:
    nombre_mostrar = integrante_sel.title()
    codigo_mostrar = str(dff["Código"].iloc[0]) if not dff.empty and "Código" in dff else ""
    st.markdown(f"## 🎓 Dashboard Estudiantil – **{nombre_mostrar}** | Código: **{codigo_mostrar}**")

# === Tabla base ===
if dff.empty:
    st.warning("No hay datos para mostrar con los filtros actuales.")
    st.stop()

st.dataframe(dff, use_container_width=True)
st.markdown("---")

# === KPIs ===
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Estudiantes", int(len(dff)))
kpi_cols[1].metric("Edad Promedio", f"{dff['Edad'].mean():.1f} años" if len(dff) else "—")
kpi_cols[2].metric("Estatura Promedio", f"{dff['Estatura_cm'].mean():.1f} cm" if len(dff) else "—")
kpi_cols[3].metric("Peso Promedio", f"{dff['Peso'].mean():.1f} kg" if len(dff) else "—")
kpi_cols[4].metric("IMC Promedio", f"{dff['IMC'].mean():.1f}" if len(dff) else "—")

st.markdown("---")

# === Gráficos ===
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Distribución por Edad")
    edad_counts = dff["Edad"].value_counts().sort_index()
    fig = plt.figure()
    edad_counts.plot(kind="bar", color="#1f77b4")
    plt.xlabel("Edad"); plt.ylabel("Cantidad"); plt.title("Distribución por Edad", color="white")
    st.pyplot(fig)

with c2:
    st.markdown("#### Distribución por Tipo de Sangre")
    rh_counts = dff["RH"].value_counts()
    fig2 = plt.figure()
    plt.pie(rh_counts.values, labels=rh_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Distribución por RH", color="white")
    st.pyplot(fig2)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Relación Estatura vs Peso")
    fig3 = plt.figure()
    plt.scatter(dff["Estatura_cm"], dff["Peso"], color="#00c3ff")
    plt.xlabel("Estatura (cm)"); plt.ylabel("Peso (kg)")
    plt.title("Estatura vs Peso", color="white")
    st.pyplot(fig3)

with c4:
    st.markdown("#### Distribución por Color de Cabello")
    cab_counts = dff["Color_Cabello"].value_counts()
    fig4 = plt.figure()
    cab_counts.plot(kind="bar", color="#ff7f0e")
    plt.xlabel("Color de Cabello"); plt.ylabel("Cantidad")
    plt.title("Distribución por Color de Cabello", color="white")
    st.pyplot(fig4)

c5, c6 = st.columns(2)
with c5:
    st.markdown("#### Distribución de Tallas de Zapatos")
    tallas = dff["Talla_Zapato"].value_counts().sort_index()
    fig5 = plt.figure()
    plt.plot(tallas.index, tallas.values, marker="o", color="#00cc99")
    plt.xlabel("Talla de Zapato"); plt.ylabel("Cantidad")
    plt.title("Distribución de Tallas de Zapatos", color="white")
    st.pyplot(fig5)

with c6:
    st.markdown("#### Top 10 Barrios de Residencia")
    top_barrios = dff["Barrio_Residencia"].value_counts().head(10)
    fig6 = plt.figure()
    top_barrios.plot(kind="bar", color="#bcbd22")
    plt.xlabel("Barrio de Residencia"); plt.ylabel("Cantidad")
    plt.title("Top 10 Barrios de Residencia", color="white")
    st.pyplot(fig6)

st.markdown("---")

# === Top 5 ===
top5_altos = dff.sort_values("Estatura_cm", ascending=False).head(5)
top5_pesados = dff.sort_values("Peso", ascending=False).head(5)

st.markdown("### 🏆 Top 5 (según filtros actuales)")
c7, c8 = st.columns(2)
with c7:
    st.write("**Mayor Estatura**")
    st.dataframe(top5_altos[["Código","Nombre_Estudiante","Apellido_Estudiante","Estatura_cm"]])
    st.download_button("Descargar CSV", data=top5_altos.to_csv(index=False).encode("utf-8"), file_name="top5_estatura.csv")

with c8:
    st.write("**Mayor Peso**")
    st.dataframe(top5_pesados[["Código","Nombre_Estudiante","Apellido_Estudiante","Peso"]])
    st.download_button("Descargar CSV", data=top5_pesados.to_csv(index=False).encode("utf-8"), file_name="top5_peso.csv")

# === Resumen ===
st.markdown("### 📈 Resumen Estadístico")
r1, r2, r3 = st.columns(3)
r1.dataframe(dff["Estatura_cm"].describe().to_frame())
r2.dataframe(dff["Peso"].describe().to_frame())
r3.dataframe(dff["IMC"].describe().to_frame())

st.caption("Clasificación IMC basada en rangos estándar (OMS).")
