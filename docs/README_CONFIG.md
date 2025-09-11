# Configuración del Sistema arXiv Papers Analysis

## Resumen de Mejoras

Se ha implementado un sistema de configuración centralizado que mejora la robustez, mantenibilidad y facilidad de uso del proyecto.

## Archivos Añadidos

### `config.py`
Sistema de configuración centralizado que:
- ✅ Centraliza todas las variables de configuración
- ✅ Proporciona validación automática
- ✅ Permite configuración por variables de entorno
- ✅ Organiza configuraciones en módulos lógicos

### `validate_config.py`
Script de validación del sistema que:
- ✅ Verifica dependencias de Python
- ✅ Valida claves API
- ✅ Confirma directorios necesarios
- ✅ Prueba inicialización de modelos
- ✅ Proporciona diagnósticos detallados

### `example_usage.py`
Ejemplos de uso del nuevo sistema de configuración.

## Configuraciones Disponibles

### DatabaseConfig
```python
vector_db_path: str = "db/chroma"
vector_collection_name: str = "arxiv_papers"
knowledge_db_dir: str = "db/knowledge"
knowledge_db_file: str = "knowledge_graph.sqlite"
```

### ModelConfig
```python
embedding_model_name: str = "all-MiniLM-L6-v2"
llm_model: str = "deepseek-chat"
chunk_size: int = 1000
chunk_overlap: int = 200
extraction_text_limit: int = 4000
```

### ArxivConfig
```python
max_results: int = 10
sort_criterion: str = "Relevance"
min_delay_seconds: int = 5
max_delay_seconds: int = 30
documents_root: str = "documentos"
```

### RAGConfig
```python
top_k_results: int = 5
similarity_threshold: float = 0.7
temperature: float = 0.1
```

### VisualizationConfig
```python
output_dir: str = "grafos"
dpi: int = 300
figure_size: tuple = (20, 20)
paper_node_color: str = "#d62728"
author_node_color: str = "#1f77b4"
```

## Uso

### 1. Validación del Sistema
```bash
python validate_config.py
```

### 2. Uso en Scripts
```python
from config import get_config

config = get_config()
print(f"Chunk size: {config.models.chunk_size}")
print(f"Vector DB path: {config.database.vector_db_path}")
```

### 3. Modificación de Configuración
```python
# Modificación temporal (solo sesión actual)
config = get_config()
config.models.chunk_size = 1500

# Modificación permanente: editar config.py o usar variables de entorno
```

### 4. Variables de Entorno
Puedes sobrescribir configuraciones usando variables de entorno en `.env`:
```bash
# Ejemplo de .env
DEEPSEEK_API_KEY=tu_clave_aquí
GOOGLE_API_KEY=tu_clave_aquí
```

## Archivos Actualizados

Todos los scripts han sido actualizados para usar la configuración centralizada:

- `get_arxiv.py` - Usa configuración de arXiv
- `rag_bbdd_vector.py` - Usa configuración de base de datos y modelos
- `ask_my_papers.py` - Usa configuración RAG
- `knowledge_graph.py` - Usa configuración de base de datos
- `analizar_investigacion.py` - Usa configuración de base de datos
- `visualizar_grafo.py` - Usa configuración de visualización
- `visu_grafo_dinamico.py` - Usa configuración de visualización

## Beneficios

### ✅ Mantenibilidad
- Todas las configuraciones en un lugar
- Fácil modificación sin tocar múltiples archivos
- Validación automática de configuraciones

### ✅ Robustez
- Validación de tipos y rangos
- Verificación de dependencias
- Manejo de errores mejorado

### ✅ Flexibilidad
- Configuración por variables de entorno
- Fácil personalización por uso específico
- Configuración modular

### ✅ Facilidad de Uso
- Script de validación automática
- Documentación integrada
- Ejemplos de uso incluidos

## Compatibilidad

Este sistema es **100% compatible** con el código existente. Todos los scripts funcionan igual que antes, pero ahora son más robustos y configurables.

## Próximos Pasos Recomendados

1. Ejecutar `python validate_config.py` antes del primer uso
2. Revisar y personalizar configuraciones en `config.py` según necesidades
3. Usar variables de entorno para configuraciones sensibles
4. Considerar implementar las mejoras adicionales propuestas