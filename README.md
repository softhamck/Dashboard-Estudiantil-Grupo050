# 🎓 Dashboard Estudiantil – Grupo 050

Dashboard interactivo desarrollado en **Python** con **Streamlit**, **Pandas** y **Matplotlib**, que permite visualizar y analizar información de los estudiantes del **Grupo 050**.

Incluye métricas, gráficos, filtros dinámicos y estadísticas descriptivas en **modo oscuro**.

---

## 🧩 Características principales

- Interfaz totalmente interactiva con **Streamlit**.
- **Modo oscuro** activado por defecto (coherente con la interfaz).
- **Filtros tolerantes**: puedes aplicar uno o varios sin necesidad de seleccionar todos.
- Cálculo automático de **edad**, **IMC** y **clasificación IMC**.
- Gráficos de barras, tortas, dispersión y línea.
- **Top 5** de mayor estatura y peso.
- Limpieza y **normalización de datos RH** (A+, o+, a positivo → A+).

---

## 📁 Estructura del proyecto

```
│
├── main.py
├── ListadoDeEstudiantesGrupo_050.xlsx
├── requirements.txt
├── README.md
└── venv/              
```

---

## 🚀 Ejecución del proyecto

Puedes ejecutarlo **de dos formas**:

1. 🐍 Usando Python instalado en tu equipo.
2. 💻 Usando Visual Studio Code (con extensiones, sin instalar Python manualmente).

---

### 🐍 Opción 1: Con Python instalado

#### 1️⃣ Requisitos previos

- Tener **Python 3.9 o superior** instalado.
- Tener **pip** actualizado:
  ```bash
  python -m pip install --upgrade pip
  ```

#### 2️⃣ Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
```

Activar el entorno:

- **Windows**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux**
  ```bash
  source venv/bin/activate
  ```

#### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4️⃣ Ejecutar el proyecto

Desde la terminal dentro del proyecto:

```bash
streamlit run main.py
```

Luego abre el enlace que aparece en la consola (por defecto):

```
http://localhost:8501
```

---

### 💻 Opción 2: Sin Python (usando Visual Studio Code)

Si no tienes Python instalado, puedes ejecutar el proyecto **solo con VS Code** gracias a las extensiones.

#### 1️⃣ Instala VS Code

Descárgalo desde [https://code.visualstudio.com/](https://code.visualstudio.com/)

#### 2️⃣ Instala las siguientes extensiones:

- 🐍 **Python** (de Microsoft)
- ⚙️ **Pylance**
- 💡 **Streamlit** *(opcional pero recomendable)*

#### 3️⃣ Abre la carpeta del proyecto

En VS Code → `File → Open Folder...` → selecciona la carpeta donde está `main.py`.

#### 4️⃣ Abre una terminal dentro de VS Code

Ve a `View → Terminal` (o usa `Ctrl + ñ`).

#### 5️⃣ Instala dependencias desde VS Code

Escribe:

```bash
pip install -r requirements.txt
```

#### 6️⃣ Ejecuta el dashboard

En la terminal integrada:

```bash
streamlit run main.py
```

El panel se abrirá automáticamente en tu navegador (por defecto en `http://localhost:8501`).

---

## 👨‍💻 Autores

- Andrea Muñoz Cano
- Camilo Andrés Fuentes Morales
- Juliana Manco Herrera
- Tomás Madrid Gómez

---

## 🪄 Licencia

Este proyecto se distribuye con fines académicos y educativos.
Puedes usarlo o modificarlo libremente citando a los autores.
