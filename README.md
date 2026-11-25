# Vision-IA: Detección de Residuos y Clasificación de Basura (Proyecto 4)

**Universidad Cooperativa de Colombia — Campus Pasto**  
**Programa:** Ingeniería de Software  
**Asignatura:** Inteligencia Artificial  
**Docente:** Ing. Cristian Camilo Ordoñez Quintero  
**Estudiante:** Anderson Ojeda  

Sistema integral de visión computacional y asistente inteligente en lenguaje natural para la detección, localización de múltiples objetos y clasificación de residuos sólidos bajo la normativa colombiana de separación en la fuente (**Resolución 2184 de 2019** del Ministerio de Ambiente).

---

## 🔒 Regla Obligatoria Cumplida: Entrenamiento 100% Desde Cero

En cumplimiento estricto con los lineamientos del proyecto:
- **Prohibido el uso de modelos preentrenados:** No se utilizan checkpoints descargados de internet, ni pesos externos de YOLO (ultralytics), ni backbones preentrenados (ResNet/VGG), ni embeddings de lenguaje preentrenados (BERT/Word2Vec/GPT), ni APIs de IA comerciales.
- **Implementación pura en PyTorch:** Tanto el detector visual multiobjeto (`WasteDetectorNet` / `VisionLocaliza-D2`) como el transformer del asistente (`WasteAssistantTransformer`) fueron inicializados con pesos aleatorios (Kaiming Normal / Xavier) y entrenados desde cero con los datos del proyecto.

---

## 🎯 Resultados Demostrados

1. **Localización Multiobjeto Simultánea:** Identifica varios residuos en una misma escena de video o fotografía. Cada objeto recibe su propio recuadro delimitador (*bounding box*), clase predicha y porcentaje de confianza.
2. **Seis Clases de Residuos:** Distingue entre:
   - `plastico` (Plástico)
   - `papel` (Papel)
   - `carton` (Cartón)
   - `vidrio` (Vidrio)
   - `metal` (Metal)
   - `organicos` (Restos Orgánicos de alimentos)
3. **Clasificación Orgánico vs. Inorgánico y Código de Colores Nacional:** Asigna automáticamente el grupo operativo y la caneca correspondiente bajo la **Resolución 2184 de 2019**:
   - ⚪ **Caneca Blanca:** Aprovechables limpios y secos (plástico, papel, cartón, vidrio, metal).
   - 🟢 **Caneca Verde:** Orgánicos aprovechables (restos de comida, desechos agrícolas).
   - ⚫ **Caneca Negra:** No aprovechables (papel higiénico, servilletas, papeles y cartones sucios/con grasa).
4. **Asistente Transformer en Español:** Responde consultas complejas sobre los residuos presentes, condiciones de suciedad/grasa, resolución de ambigüedad cuando hay múltiples objetos, y descarte de preguntas fuera de alcance.
5. **Modo Comparativo Lado a Lado:** Compara las predicciones del modelo frente a las anotaciones humanas de referencia (*Ground Truth*) evaluando la métrica de solapamiento $IoU \ge 0.5$ con emparejamiento uno a uno.

---

## 🚀 Guía Rápida de Comandos en Windows PowerShell

Todos los comandos son reproducibles en la terminal de PowerShell desde la raíz del proyecto:

### 1. Activar Entorno y Verificar Dependencias
```powershell
# Usando Python del sistema o Anaconda:
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m pip install torch torchvision opencv-python-headless streamlit pillow numpy
```

### 2. Anotar Nuevos Videos o Secuencias de Fotos
Abre la herramienta interactiva de anotación con el mouse:
```powershell
python -m scripts.annotate_tool --source ruta/al/video.mp4 --session sesion_01
```
*Atajos:* Clic y arrastrar para trazar recuadro. Teclas `1`-`6` para elegir clase. Tecla `d` para borrar última caja. Tecla `e` para marcar fotograma vacío (ejemplo negativo sin residuos). `s` para guardar, `n` siguiente, `q` salir.

### 3. Auditar Dataset y Particiones (Anti-Leakage)
Informa la cantidad de videos, imágenes pendientes, fotogramas vacíos revisados y conteo de objetos por clase y partición (*train, val, test*):
```powershell
python -m scripts.data_summary
```

### 4. Entrenar el Detector Multiobjeto Desde Cero
Entrena la red convolucional `WasteDetectorNet` con pesos aleatorios, optimizador AdamW, guarda el mejor checkpoint y evalúa con $IoU \ge 0.5$:
```powershell
python -m scripts.train_waste_detector --epochs 15 --batch_size 8 --lr 0.001
```
*(Para verificar rápidamente el flujo con datos de prueba técnica: `python -m scripts.train_waste_detector --technical_check --epochs 3`)*

### 5. Entrenar el Transformer del Asistente en Español
Construye el vocabulario desde cero y entrena el Transformer Encoder en PyTorch:
```powershell
python -m scripts.train_assistant_transformer --epochs 35 --lr 0.001
```

### 6. Ejecutar Pruebas Unitarias Automatizadas
Verifica que el almacenamiento conserve exactamente las 3 cajas y clases de una imagen, y comprueba el motor de ambigüedades:
```powershell
python -m unittest tests.test_annotation_system tests.test_assistant_transformer
```

### 7. Iniciar la Aplicación Interactiva Local
Abre la interfaz gráfica en el navegador (funciona sin conexión a internet):
```powershell
streamlit run app/main.py
```
Abre automáticamente en: `http://localhost:8501`.

---

## 📁 Estructura del Proyecto

```
IA-Vision/
│── app/                            # Aplicación interactiva Streamlit
│   ├── main.py                     # Interfaz gráfica principal con 5 pestañas
│   ├── detector.py                 # Inferencia multiobjeto y recuadros de color
│   ├── recycling.py                # Reglas base de separación y colores Colombia
│   ├── resources.py                # Carga en caché del detector y transformer
│   └── camera_component/           # Componente nativo de video en navegador
│── docs/                           # Documentación académica y técnica
│   ├── INFORME_PROYECTO_FINAL.md   # Informe técnico formal con metodología y métricas
│   ├── GUIA_SUSTENTACION_10_MINUTOS.md # Guion de 10 min y banco de preguntas para el profesor
│   ├── GUIA_CAPTURA_ETIQUETADO.md  # Protocolo de grabación y partición anti-leakage
│   ├── RECYCLING_KNOWLEDGE.md      # Base normativa: Resolución 2184 de 2019
│   └── DATASETS.md                 # Análisis y licencias de datasets públicos
│── models/runtime/                 # Checkpoints propios entrenados en el proyecto
│   ├── detector_D2.pth             # Detector convolucional propio D2
│   ├── assistant_transformer.pth   # Transformer del asistente entrenado desde cero
│   ├── assistant_vocab.json        # Vocabulario léxico del asistente
│   └── manifest.json               # Manifiesto de procedencia estricta
│── scripts/                        # Comandos ejecutables en PowerShell
│   ├── annotate_tool.py            # Herramienta gráfica de anotación de video
│   ├── data_summary.py             # Auditoría de videos, imágenes y objetos
│   ├── train_waste_detector.py     # Entrenamiento y evaluación IoU del detector
│   └── train_assistant_transformer.py # Entrenamiento del transformer en español
│── tests/                          # Pruebas unitarias automatizadas
│   ├── test_annotation_system.py   # Prueba de guardado/reapertura de 3 residuos
│   └── test_assistant_transformer.py # Prueba de intenciones y ambigüedad
│── training/                       # Arquitecturas y módulos de red
│   ├── multi_waste_detector.py     # WasteDetectorNet y evaluador uno a uno IoU
│   ├── assistant_transformer.py    # WasteAssistantTransformer y motor normativo
│   └── assistant_data.py           # Corpus de entrenamiento y test desacoplado
└── utils/                          # Utilidades auxiliares
    ├── annotation_storage.py       # Gestor JSON y partición por sesiones
    └── visual_comparison.py        # Generador de comparaciones lado a lado
```

---

## 🏛️ Créditos y Autoría
- **Estudiante:** Anderson Ojeda — Ingeniería de Software
- **Docente:** Ing. Cristian Camilo Ordoñez Quintero — Inteligencia Artificial
- **Universidad Cooperativa de Colombia** — Sede Pasto (2026)
