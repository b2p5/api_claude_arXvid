# arXiv Papers Analysis System - Web Interface

Una interfaz web completa construida con Streamlit para el sistema de análisis de papers académicos de arXiv.

## 🚀 Características Principales

### 📊 Dashboard Interactivo
- **Métricas del sistema** en tiempo real
- **Visualizaciones de corpus** con gráficos interactivos
- **Estadísticas de análisis** de contenido
- **Estado del sistema** y monitoreo

### 📄 Análisis de Papers
- **Búsqueda avanzada** (semántica, por palabras clave, por título)
- **Upload de PDFs** para análisis inmediato
- **Análisis detallado** con referencias, conceptos y temas
- **Visualización de secciones** con resúmenes automáticos

### 🔍 Análisis de Contenido Avanzado
- **Co-ocurrencia de conceptos** con visualización de red
- **Análisis de red de citas** con métricas de centralidad
- **Evolución de temas** a lo largo del tiempo
- **Tendencias técnicas** y niveles de complejidad

### 🕸️ Grafo de Conocimiento
- **Visualización interactiva** de redes de citación
- **Relaciones entre conceptos** con diferentes algoritmos de layout
- **Conexiones entre temas** y colaboraciones
- **Grafos personalizables** con filtros avanzados

### 📤 Export y Reportes
- **Múltiples formatos** (JSON, CSV, Excel)
- **Filtros temporales** y por características técnicas
- **Reportes pre-construidos** del sistema
- **Análisis completos** exportables

### ⚙️ Configuración del Sistema
- **Ajustes de análisis** y procesamiento
- **Configuración de visualizaciones**
- **Gestión de base de datos**
- **Diagnósticos del sistema**

## 🔧 Instalación y Configuración

### Requisitos Previos
- Python 3.8+
- Sistema de análisis de arXiv configurado
- Base de datos con papers procesados

### Instalación Automática
```bash
# Usar el launcher que instala dependencias automáticamente
python launch_web_interface.py
```

### Instalación Manual
```bash
# Instalar dependencias web
pip install -r requirements_web.txt

# Instalar modelo de spaCy (opcional)
python -m spacy download en_core_web_sm

# Lanzar interfaz
streamlit run web_interface.py
```

### Dependencias Principales
- **streamlit** >= 1.28.0 - Framework web
- **plotly** >= 5.15.0 - Visualizaciones interactivas
- **pandas** >= 2.0.0 - Manipulación de datos
- **networkx** >= 3.0 - Análisis de redes
- **scikit-learn** >= 1.3.0 - Análisis avanzado

## 🎮 Uso de la Interfaz

### 1. Página Principal (Dashboard)
- **Métricas generales** del sistema
- **Gráficos de conceptos** más frecuentes
- **Distribución técnica** de papers
- **Actividad reciente** de análisis

### 2. Análisis de Papers
```
🔍 Búsqueda → Selección → 📊 Análisis Detallado
```
- Buscar papers por diferentes criterios
- Seleccionar paper para análisis completo
- Ver referencias, conceptos, temas y secciones
- Exportar resultados individuales

### 3. Análisis de Contenido Avanzado
```
🎛️ Opciones → 🔄 Análisis → 📊 Visualización
```
- Seleccionar tipo de análisis
- Configurar filtros temporales
- Generar visualizaciones interactivas
- Explorar patrones y tendencias

### 4. Grafo de Conocimiento
```
🕸️ Tipo de Grafo → 🎨 Layout → 📊 Visualización
```
- Elegir tipo de red (citación, conceptos, temas)
- Seleccionar algoritmo de layout
- Ajustar número de nodos
- Interactuar con la visualización

### 5. Export y Reportes
```
📊 Tipo de Datos → 📁 Formato → 🕒 Filtros → 📥 Descarga
```
- Seleccionar datos a exportar
- Elegir formato de salida
- Aplicar filtros temporales y técnicos
- Descargar archivos generados

## 📊 Tipos de Visualizaciones

### Gráficos de Barras
- Conceptos más frecuentes
- Papers por nivel técnico
- Actividad de citación

### Gráficos Circulares
- Distribución de niveles técnicos
- Categorías de temas

### Gráficos de Dispersión
- Relación frecuencia vs importancia de conceptos
- Análisis de temas por peso

### Redes Interactivas
- Co-ocurrencia de conceptos
- Redes de citación
- Conexiones entre temas

### Series Temporales
- Evolución de temas
- Tendencias de complejidad técnica

## 🔍 Funcionalidades de Búsqueda

### Búsqueda Semántica
- Utiliza embeddings para encontrar papers similares
- Basada en el contenido y contexto
- Ranking por relevancia semántica

### Búsqueda por Palabras Clave
- Búsqueda directa en título y resumen
- Soporte para términos técnicos
- Filtrado por coincidencias exactas

### Búsqueda por Título
- Búsqueda específica en títulos
- Soporte para búsquedas parciales
- Ideal para papers conocidos

## 📤 Formatos de Export

### JSON
- Estructura completa de datos
- Ideal para procesamiento programático
- Incluye metadata y filtros aplicados

### CSV
- Formato tabular para análisis estadístico
- Compatible con Excel y herramientas de análisis
- Datos principales en formato plano

### Excel
- Múltiples hojas por tipo de dato
- Formato profesional para reportes
- Incluye papers, conceptos, referencias, temas

## 🛠️ Configuración Avanzada

### Variables de Entorno
```bash
# Puerto personalizado
STREAMLIT_SERVER_PORT=8501

# Tema de la interfaz
STREAMLIT_THEME_BASE="light"

# Configuración de análisis
ENABLE_CONTENT_ANALYSIS=true
MAX_PROCESSING_WORKERS=4
```

### Configuración del Sistema
- **Análisis de contenido**: Habilitar/deshabilitar análisis profundo
- **Workers paralelos**: Número de procesos para análisis
- **Tamaño de chunks**: Configuración de procesamiento de texto
- **Límite de nodos**: Número máximo de nodos en grafos

## 🔧 Diagnósticos y Mantenimiento

### Diagnósticos Automáticos
- **Conexión a base de datos**
- **Módulos requeridos**
- **Procesador RAG**
- **Análisis de contenido**
- **Dependencias Python**

### Mantenimiento
```bash
# Actualizar estadísticas
🔄 Refresh Statistics

# Limpiar caché
🧹 Clean Cache  

# Reconstruir índices
📊 Rebuild Indexes
```

## 🚨 Solución de Problemas

### Error: "Module not found"
```bash
# Instalar dependencias faltantes
pip install -r requirements_web.txt
```

### Error: "Database not found"
```bash
# Ejecutar análisis principal primero
python rag_bbdd_vector_optimized.py --input-dir /path/to/pdfs
```

### Error: "Port already in use"
```bash
# Cambiar puerto
streamlit run web_interface.py --server.port 8502
```

### Error: "Memory issues"
- Reducir número de workers paralelos
- Disminuir tamaño de chunks
- Limitar nodos en visualizaciones de red

## 📝 Logs y Monitoreo

### Ubicación de Logs
- Logs del sistema: `logs/`
- Logs de análisis: Integrados en interfaz
- Logs de Streamlit: Terminal/consola

### Niveles de Log
- **INFO**: Operaciones normales
- **WARNING**: Problemas no críticos  
- **ERROR**: Errores que requieren atención

## 🔄 Actualizaciones

### Actualizar Dependencias
```bash
pip install --upgrade -r requirements_web.txt
```

### Actualizar Interfaz
```bash
git pull origin main
streamlit run web_interface.py --server.runOnSave true
```

## 📞 Soporte

Para problemas específicos de la interfaz web:
1. Verificar diagnósticos del sistema
2. Revisar logs de error
3. Comprobar dependencias
4. Consultar este README

La interfaz web proporciona una experiencia completa e interactiva para explorar y analizar el corpus de papers académicos procesados por el sistema.