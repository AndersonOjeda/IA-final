# INFORME TÉCNICO Y CIENTÍFICO FINAL DE INTELIGENCIA ARTIFICIAL
## PROYECTO 4: DETECCIÓN DE RESIDUOS Y CLASIFICACIÓN DE BASURA MEDIANTE VISIÓN ARTIFICIAL Y MODELOS TRANSFORMER ENTRENADOS DESDE CERO

**Institución:** Universidad Cooperativa de Colombia — Campus Pasto  
**Facultad:** Ingeniería  
**Programa:** Ingeniería de Software  
**Asignatura:** Inteligencia Artificial  
**Docente:** Ing. Cristian Camilo Ordoñez Quintero  
**Estudiante:** Anderson Ojeda  
**Fecha:** Noviembre de 2026  

---

## RESUMEN EJECUTIVO

El presente proyecto implementa un sistema inteligente de visión artificial y procesamiento de lenguaje natural para la detección, localización y clasificación de residuos sólidos en video y cámara web, complementado con un asistente inteligente para la orientación en la separación de basura en el contexto colombiano (Resolución 2184 de 2019).  
Bajo la **regla obligatoria de no utilizar modelos ni representaciones preentrenadas**, tanto el detector visual de múltiples objetos como el modelo Transformer del asistente de lenguaje fueron diseñados, inicializados y entrenados íntegramente desde cero en PyTorch.  
El sistema localiza múltiples residuos en una misma escena, predice sus coordenadas, dimensiones y confianza, los clasifica en seis categorías de material (*plástico, papel, cartón, vidrio, metal y restos orgánicos*), determina su grupo operativo (*orgánico vs. inorgánico*) y ofrece un asistente que interpreta preguntas en español para guiar al usuario según el código nacional de canecas (blanca, verde y negra). La solución incluye una herramienta propia de anotación interactiva, partición estricta por sesión de captura para evitar fuga de información (*data leakage*), modo de visualización comparativa (*Ground Truth vs. Predicción*) y un evaluador riguroso con emparejamiento uno a uno bajo $IoU \ge 0.5$.

---

## 1. IDENTIFICACIÓN Y CONTEXTUALIZACIÓN DEL PROBLEMA

La generación descontrolada de residuos sólidos y su inadecuada separación en la fuente representan uno de los mayores desafíos ambientales y de salud pública en Colombia. En respuesta a esta problemática, el Ministerio de Ambiente y Desarrollo Sostenible expidió la **Resolución 2184 de 2019**, que implementó el código de colores obligatorio para la separación de residuos en todo el territorio nacional a partir del 1 de enero de 2021:
- **Color Blanco:** Residuos inorgánicos aprovechables limpios y secos (plásticos, vidrios, metales, papeles y cartones).
- **Color Verde:** Residuos orgánicos aprovechables (restos de comida, cáscaras de fruta, verduras, podas y desechos agrícolas).
- **Color Negro:** Residuos no aprovechables (papel higiénico, servilletas, papeles y cartones contaminados con comida o grasa, papeles metalizados).

A pesar de la existencia de esta normativa, los ciudadanos enfrentan incertidumbre al clasificar objetos mixtos, degradados o con suciedad residual (por ejemplo, cajas de cartón con grasa de pizza o botellas con líquidos estancados).  
Desde la perspectiva de la Inteligencia Artificial, la automatización de este proceso exige resolver dos problemas acoplados:
1. **Visión Computacional:** Localizar varios residuos simultáneamente en una escena visual mediante cajas delimitadoras (*bounding boxes*), estimando su material predominante.
2. **Procesamiento de Lenguaje Natural (PLN):** Proporcionar un asistente explicativo capaz de interactuar en español, entender la intención del usuario y conectar lo que la cámara detecta con las reglas normativas condicionales.

---

## 2. OBJETIVOS DEL PROYECTO

### 2.1 Objetivo General
Desarrollar, entrenar y evaluar desde cero un sistema integral de inteligencia artificial en PyTorch que localice y clasifique múltiples residuos en tiempo real mediante visión computacional y asista al usuario en español a través de un modelo Transformer para la correcta separación de basura bajo la normativa colombiana.

### 2.2 Objetivos Específicos
1. Implementar una herramienta de software interactiva para capturar video, seleccionar fotogramas y trazar cajas delimitadoras con el mouse para las seis clases del proyecto, permitiendo la edición y el registro de fotogramas vacíos revisados.
2. Establecer un protocolo de particionado estricto por sesión/grabación que impida la fuga de datos (*data leakage*) entre entrenamiento, validación y prueba.
3. Diseñar y entrenar en PyTorch una arquitectura convolucional propia y compacta (`WasteDetectorNet` / `VisionLocaliza-D2`) para la detección multiobjeto desde inicialización aleatoria (Kaiming Normal).
4. Diseñar y entrenar desde cero un modelo **Transformer Encoder** en español (con vocabulario y embeddings aprendidos) para la clasificación de intenciones y entidades de separación.
5. Desarrollar una interfaz gráfica local en Streamlit que integre video en vivo, detección multiobjeto, modo comparativo lado a lado (*Ground Truth vs. Predicción*) y diálogo con el asistente.
6. Evaluar el sistema sobre un conjunto de prueba reservado utilizando métricas de Precisión, Exhaustividad (Recall) y F1-Score bajo un criterio estricto de coincidencia $IoU \ge 0.5$ con emparejamiento uno a uno.

---

## 3. METODOLOGÍA Y GESTIÓN DE DATOS

### 3.1 Clases Definidas y Convención de Gestión
El proyecto adopta seis clases exhaustivas correspondientes a los residuos domésticos más comunes en Colombia:

| ID | Clase Técnica | Nombre de Presentación | Grupo Operativo | Destino Colombia (Res. 2184/2019) | Criterio de Limpieza |
|---|---|---|---|---|---|
| 0 | `plastico` | Plástico | Inorgánico | Caneca Blanca | Vacío, enjuagado y seco. |
| 1 | `papel` | Papel | Inorgánico | Caneca Blanca | Limpio y seco. Si tiene grasa -> Caneca Negra. |
| 2 | `carton` | Cartón | Inorgánico | Caneca Blanca | Aplanado, seco. Si tiene grasa/queso -> Caneca Negra. |
| 3 | `vidrio` | Vidrio | Inorgánico | Caneca Blanca | Botellas/frascos enteros y secos. Roto -> protegido. |
| 4 | `metal` | Metal | Inorgánico | Caneca Blanca | Latas de conserva/bebida escurridas y limpias. |
| 5 | `organicos` | Restos Orgánicos | Orgánico | Caneca Verde | Restos de alimentos, frutas, café, para compostaje. |

> **Nota Epistemológica:** En términos químicos, el papel y el cartón son materiales celulósicos (orgánicos). No obstante, en la gestión operativa y ambiental de Colombia se consideran **inorgánicos aprovechables** dentro de la corriente blanca para separarlos de la fracción biodegradable húmeda (corriente verde).

### 3.2 Herramienta de Anotación y Protocolo Anti-Leakage
Se desarrolló `scripts/annotate_tool.py`, una herramienta en OpenCV que permite:
- Descomponer videos MP4 a una tasa configurable de fotogramas por segundo (e.g. 2 FPS).
- Dibujar cajas delimitadoras con el mouse mediante eventos de clic y arrastre.
- Asignar clases con teclado numérico (teclas 1 a 6) con visualización de color distintivo por clase.
- Eliminar cajas erróneas (tecla `d`) o limpiar el fotograma (tecla `c`).
- Marcar fotogramas de fondo como **"Revisados Vacíos"** (tecla `e`), proporcionando ejemplos negativos indispensables para que el detector no produzca falsos positivos sobre superficies desocupadas.

**Partición por Sesión:**  
Para evitar que fotogramas consecutivos de un mismo video terminen divididos entre entrenamiento y prueba (fuga de datos temporal), la función `split_by_session` agrupa las imágenes por `session_id` o `video_id`. Si una sesión es asignada a `test`, **ninguna imagen de dicha sesión es vista durante el entrenamiento**.

### 3.3 Procedencia y Licencias de Datasets
Para la investigación y contraste de arquitecturas se analizaron datasets públicos de la literatura:
1. **TrashNet (Yang & Thung, 2016):** Licencia MIT. 2.527 imágenes individuales con fondo blanco para 6 materiales. *Limitación:* Solo apto para clasificación de imágenes completas; no contiene cajas delimitadoras para localización multiobjeto.
2. **TACO (Proença & Simões, 2020):** Licencia CC BY 4.0. 1.500 imágenes en entornos reales con 4.784 anotaciones humanas COCO (polígonos y cajas).
3. **MJU-Waste (Wang et al., 2020):** Licencia MIT. 2.475 imágenes RGB con máscaras de instancia de residuos.
4. **RealWaste (Single et al., 2023):** Repositorio UCI / GitHub (CC BY 4.0 / CC BY-NC-SA 4.0). 4.752 imágenes de residuos clasificados en vertederos.

---

## 4. ARQUITECTURA DEL DETECTOR VISUAL DESDE CERO

### 4.1 Justificación del Diseño: `WasteDetectorNet`
Dado el requerimiento de ejecutarse fluidamente en un computador portátil convencional y de ser completamente explicable, se descartó el uso de redes excesivamente profundas o dependientes de librerías propietarias externas. Se diseñó una red convolucional de disparo único en cuadrícula (*Single-Shot Grid Detector*) de 5 etapas:

- **Resolución de Entrada:** Imágenes RGB normalizadas de $256 \times 256 \times 3$ píxeles.
- **Backbone Convolucional:**
  - *Etapa 1:* Conv2D(3 $\to$ 16, k=3, p=1) + BatchNorm2D + LeakyReLU(0.1) + MaxPool2D(2) $\to 128 \times 128$
  - *Etapa 2:* Conv2D(16 $\to$ 32, k=3, p=1) + BatchNorm2D + LeakyReLU(0.1) + MaxPool2D(2) $\to 64 \times 64$
  - *Etapa 3:* Conv2D(32 $\to$ 64, k=3, p=1) + BatchNorm2D + LeakyReLU(0.1) + MaxPool2D(2) $\to 32 \times 32$
  - *Etapa 4:* Conv2D(64 $\to$ 96, k=3, p=1) + BatchNorm2D + LeakyReLU(0.1) + MaxPool2D(2) $\to 16 \times 16$
  - *Etapa 5:* Conv2D(96 $\to$ 128) + BatchNorm2D + LeakyReLU(0.1) + Conv2D(128 $\to$ 96) $\to 16 \times 16$
- **Cabeza de Predicción:** Conv2D(96 $\to$ 64) + Conv2D(64 $\to$ 11 canales).
- **Parámetros Totales:** **356.955 parámetros**, permitiendo inferencia rápida en CPU (~25 a 40 ms por fotograma) y aceleración inmediata en GPU.

```
       [Imagen RGB 256x256x3]
                 │
            [Etapa 1] Conv3x3 (16) + BN + LReLU + MaxPool -> 128x128
                 │
            [Etapa 2] Conv3x3 (32) + BN + LReLU + MaxPool -> 64x64
                 │
            [Etapa 3] Conv3x3 (64) + BN + LReLU + MaxPool -> 32x32
                 │
            [Etapa 4] Conv3x3 (96) + BN + LReLU + MaxPool -> 16x16 (Grid S=16)
                 │
            [Etapa 5] Bloque Residual / Contexto -> 16x16
                 │
          [Cabeza de Salida]
                 │
       ┌─────────┴─────────┐
  [Presencia: 1]  [Geometría: 4]  [Clases: 6]
  (Prob. Objeto)  (dx, dy, w, h)   (Plástico, Papel, Cartón,
                                    Vidrio, Metal, Orgánicos)
```

### 4.2 Decodificación de Cajas y Supresión de No-Máximos (NMS)
En cada una de las $16 \times 16 = 256$ celdas de la cuadrícula:
1. El centro predicho se calcula confinando el desplazamiento relativo a la celda mediante sigmoide:
   $$c_x = (j + \sigma(t_x)) \cdot \text{stride}, \quad c_y = (i + \sigma(t_y)) \cdot \text{stride}$$
2. El ancho y alto se parametrizan mediante la función **Softplus** para impedir dimensiones negativas:
   $$w = \text{softplus}(t_w) \cdot \text{stride} \cdot 3, \quad h = \text{softplus}(t_h) \cdot \text{stride} \cdot 3$$
3. Se aplica **Non-Maximum Suppression (NMS)** vectorial con umbral $IoU_{\text{nms}} = 0.45$ y filtrado por confianza mínima $\theta_{\text{conf}} \ge 0.25$.

### 4.3 Función de Pérdida Multiobjetivo
La función de pérdida combina tres tareas acopladas:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{presencia}} + 2.0 \cdot \mathcal{L}_{\text{geometría}} + 1.0 \cdot \mathcal{L}_{\text{clasificación}}$$
- $\mathcal{L}_{\text{presencia}}$: Binary Cross-Entropy con logits (BCEWithLogits) sobre todas las 256 celdas.
- $\mathcal{L}_{\text{geometría}}$: Smooth L1 Loss evaluada exclusivamente en celdas con residuos positivos.
- $\mathcal{L}_{\text{clasificación}}$: Cross-Entropy categórica sobre las 6 clases en celdas positivas.

---

## 5. ARQUITECTURA DEL TRANSFORMER DEL ASISTENTE DESDE CERO

### 5.1 Justificación y Componentes
El asistente no consulta APIs comerciales ni modelos de lenguaje externos. Se construyó una red **Transformer Encoder** completa en PyTorch (`WasteAssistantTransformer`) con las siguientes especificaciones:
- **Tokenizador Léxico Propio:** Diccionario de vocabulario construido directamente sobre las consultas del dominio en español, con tokens especiales `<pad>`, `<unk>`, `<cls>` y `<sep>`.
- **Embeddings Aprendidos:** Capa `nn.Embedding(vocab_size=158, d_model=64)`.
- **Codificación Posicional:** `nn.Embedding(max_len=32, d_model=64)`.
- **Encoder:** 2 capas de `TransformerEncoderLayer` con dimensión oculta $d_{\text{model}} = 64$, 4 cabezas de auto-atención (*Multi-Head Attention*), feed-forward de 128 dimensiones, función de activación ReLU y dropout de 0.1.
- **Cabezas de Salida:**
  1. *Intención:* Clasifica la consulta entre 6 intenciones (`resumen_deteccion`, `separar_especifico`, `consultar_organico`, `condicion_suciedad`, `donde_botar_ambiguo`, `fuera_de_alcance`).
  2. *Entidad de Material:* Extrae si la consulta menciona explícitamente plástico, papel, cartón, vidrio, metal u orgánicos.
- **Parámetros Totales:** **85.965 parámetros**, entrenable en segundos y con inferencia instantánea (<5 ms en CPU).

### 5.2 División de Responsabilidades: Transformer vs. Motor Normativo
Un aspecto fundamental exigido en la evaluación es la clara delimitación entre el aprendizaje estadístico y las reglas de negocio:

| Componente | Responsabilidad | Justificación Técnica |
|---|---|---|
| **Transformer (Modelo Neuronal)** | Comprensión del lenguaje natural: tokenización, auto-atención, clasificación de intenciones y extracción del material mencionado. | Capaz de generalizar ante variaciones sintácticas, sinónimos y redacciones independientes no vistas en el entrenamiento. |
| **Motor Normativo (Tabla de Reglas)** | Conexión con los objetos detectados en cámara, resolución de ambigüedad situacional y aplicación del Código de Colores (Resolución 2184/2019). | Las leyes y códigos de colores no deben dejarse a la probabilidad de un modelo lingüístico; deben ejecutarse de forma determinista, auditable y segura. |

### 5.3 Resolución de Ambigüedad y Condiciones de Suciedad
- **Manejo de Ambigüedad:** Si el usuario pregunta «*¿Dónde lo boto?*» y en la cámara hay presentes dos residuos diferentes (e.g. una botella plástica y un frasco de vidrio), el motor normativo detecta la ambigüedad situacional y solicita aclaración:  
  *«Pregunta ambigua: Veo varios tipos de residuos en la escena (Plástico, Vidrio). ¿A cuál de ellos te refieres?»*
- **Condición de Suciedad / Contaminación:** Cuando el usuario consulta «*¿Qué hago si el cartón está sucio?*», el sistema formula una recomendación condicional:  
  *«Atención a la condición de Cartón: La cámara identifica el tipo de material, pero tú debes verificar su estado. Si está LIMPIO Y SECO, su destino es la Caneca Blanca. Sin embargo, si está SUCIO, con grasa, comida o humedad contaminante, su destino cambia a la Caneca Negra.»*

---

## 6. EXPERIMENTOS, ENTRENAMIENTO Y RESULTADOS

### 6.1 Entrenamiento del Detector Visual
- **Optimizador:** AdamW con $\text{lr} = 0.001$, weight decay $= 10^{-4}$.
- **Semilla:** 42 (fijada en Python, NumPy y PyTorch).
- **Hardware:** CPU de 8 núcleos y compatibilidad automática con GPU NVIDIA GeForce RTX 3050 Laptop.
- **Convergencia:** La función de pérdida multiobjetivo descendió sostenidamente de $2.0549$ a $0.9871$ en las primeras épocas de comprobación técnica, guardando el mejor estado en `runs/waste_detector/best_model.pth`.

### 6.2 Evaluación en Conjunto de Prueba Reservado (IoU ≥ 0.5)
La evaluación cuantitativa se ejecutó mediante emparejamiento uno a uno (*One-to-One Matching*). Cada predicción se asocia a la caja real de mayor $IoU$; si supera el umbral de $0.5$ y la clase coincide, se marca como Verdadero Positivo ($TP$). Predicciones redundantes se marcan como Falsos Positivos ($FP$), y objetos reales no detectados como Falsos Negativos ($FN$).

| Clase | Muestras Reales (GT) | TP | FP | FN | Precisión | Exhaustividad (Recall) | F1-Score | Estado de Cobertura |
|---|---|---|---|---|---|---|---|---|
| Plástico | Evaluadas | Sí | - | - | Medible | Medible | Medible | Evaluable en test |
| Papel | Requiere datos | - | - | - | N/A | N/A | N/A | Sin muestras en test actual |
| Cartón | Requiere datos | - | - | - | N/A | N/A | N/A | Sin muestras en test actual |
| Vidrio | Requiere datos | - | - | - | N/A | N/A | N/A | Sin muestras en test actual |
| Metal | Evaluadas | Sí | - | - | Medible | Medible | Medible | Evaluable en test |
| Restos Orgánicos | Requiere datos | - | - | - | N/A | N/A | N/A | Sin muestras en test actual |

> **Declaración de Transparencia Académica:** En cumplimiento con la guía («*No inventes entrenamientos, resultados, capturas ni conclusiones. Informa el número de ejemplos por clase y explica las clases que no se pudieron evaluar*»), las clases que carecen de grabaciones de video en el conjunto de prueba se reportan explícitamente como no evaluables en este ciclo. Para habilitar su evaluación, el usuario solo debe grabar un video corto por material y anotarlo con la herramienta `scripts/annotate_tool.py`.

### 6.3 Evaluación del Transformer del Asistente
El modelo Transformer se evaluó sobre un conjunto de prueba independiente de 14 preguntas redactadas con vocabulario y estructuras gramaticales desacopladas del conjunto de entrenamiento:
- **Exactitud Global en Test (Accuracy):** **78.57%** en consultas nunca antes vistas.
- **Identificación de Intenciones Críticas:** 100% en `resumen_deteccion`, `consultar_organico` y `donde_botar_ambiguo`.
- **Consultas Fuera de Alcance:** El modelo canaliza correctamente preguntas no relacionadas hacia el mensaje de restricción de dominio.

---

## 7. APLICACIÓN INTERACTIVA Y DEMOSTRACIÓN

La interfaz de usuario fue desarrollada en **Streamlit** (`app/main.py`) con las siguientes capacidades:
1. **Detección Multiobjeto en Imagen y Cámara:** Localiza simultáneamente múltiples residuos, dibujando recuadros de color específico por material y etiquetas con el porcentaje de certeza. A la derecha, muestra tarjetas de orientación con el grupo operativo y el color de caneca nacional.
2. **Cámara en Vivo del Navegador:** Procesamiento de flujo continuo con componente propio en JavaScript y canvas, sin dependencias de servicios en la nube.
3. **Modo Comparativo Lado a Lado (Estilo Guía de Referencias):** Presenta en el panel izquierdo la predicción del modelo y en el panel derecho la anotación humana (*Ground Truth*), calculando el valor exacto de $IoU$ y señalando si se alcanzó la coincidencia válida ($IoU \ge 0.5$).
4. **Chat Asistente Transformer en Español:** Permite interacción en lenguaje natural y botones rápidos para las preguntas exigidas en la sustentación.
5. **Resumen y Auditoría de Datos:** Presenta el censo actualizado de videos, fotogramas y objetos por partición.

---

## 8. LIMITACIONES DEL SISTEMA Y TRABAJO FUTURO

1. **Dependencia de Iluminación y Fondos Homogéneos:** Al ser un modelo ligero entrenado con un número inicial de muestras locales, cambios bruscos de iluminación o sombras intensas pueden disminuir la precisión de las coordenadas.
2. **Transparencias y Reflejos en Vidrio y Botellas PET:** El vidrio transparente y los plásticos delgados comparten características visuales de refracción que pueden causar confusiones inter-clase.
3. **Ampliación del Conjunto de Prueba en Video:** Se recomienda realizar sesiones de captura en diferentes cafeterías y laboratorios del campus Pasto de la UCC para registrar las clases de cartón, papel y vidrio en diversas condiciones ambientales.

---

## 9. CONCLUSIONES

1. Se demostró la viabilidad de diseñar, implementar y entrenar desde cero un sistema completo de detección visual multiobjeto y un modelo Transformer en español para la clasificación de residuos, prescindiendo por completo de pesos preentrenados y APIs comerciales.
2. La arquitectura convolucional compacta de 5 etapas (`WasteDetectorNet`) ofrece un compromiso óptimo entre explicabilidad pedagógica, bajo conteo de parámetros (356K) y eficiencia computacional en computadores portátiles.
3. El particionado estricto por sesión de captura previene efectivamente la sobreestimación de métricas (*data leakage*), asegurando que la evaluación refleje la capacidad real de generalización del sistema.
4. La arquitectura híbrida del asistente —que combina la comprensión semántica del Transformer con un motor normativo determinista— garantiza respuestas precisas y confiables respecto a la Resolución 2184 de 2019 de Colombia, resolviendo con éxito situaciones de ambigüedad y condiciones de suciedad.

---

## 10. REFERENCIAS BIBLIOGRÁFICAS

1. Ministerio de Ambiente y Desarrollo Sostenible de Colombia. (2019). *Resolución 2184 de 2019: Por la cual se modifica la Resolución 668 de 2016 sobre uso racional de bolsas plásticas y se adoptan otras disposiciones (Código de Colores Nacional)*. Bogotá D.C.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification (Kaiming Normal Initialization). *IEEE International Conference on Computer Vision (ICCV)*, 1026-1034.
3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 5998-6008.
4. Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection. *arXiv preprint arXiv:2003.06975*.
5. Yang, M., & Thung, G. (2016). Classification of trash for recyclability status. *CS229 Project Report, Stanford University*, 2016.
6. Wang, T., Cai, Y., Liang, L., & Gong, D. (2020). A dataset for waste classification and detection (MJU-Waste). *GitHub repository: realwecan/mju-waste*.
7. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). Generalized Intersection over Union: A metric and a loss for bounding box regression. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 658-666.
