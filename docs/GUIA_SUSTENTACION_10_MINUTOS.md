# GUÍA Y GUION DE SUSTENTACIÓN FINAL (10 MINUTOS)
## PROYECTO 4: DETECCIÓN DE RESIDUOS Y CLASIFICACIÓN DE BASURA

**Institución:** Universidad Cooperativa de Colombia — Campus Pasto  
**Programa:** Ingeniería de Software  
**Asignatura:** Inteligencia Artificial  
**Docente:** Ing. Cristian Camilo Ordoñez Quintero  
**Estudiante:** Anderson Ojeda  

---

## 1. Distribución del Tiempo de la Sustentación (10 Minutos)

| Minuto | Sección | Contenido Clave | Apoyo Visual |
|---|---|---|---|
| **0:00 - 1:30** | **Introducción y Problema** | Contexto de separación en la fuente en Colombia (Res. 2184/2019), reto de localizar múltiples residuos en escena sin modelos preentrenados. | Diapositiva 1-2 |
| **1:30 - 3:30** | **Metodología y Datos** | Protocolo de captura en video, herramienta propia de anotación, partición por sesión (anti-leakage), fotogramas vacíos revisados. | Diapositiva 3-4 |
| **3:30 - 5:30** | **Arquitectura y Entrenamiento desde Cero** | Red convolucional compacta de 5 etapas (`WasteDetectorNet` / `VisionLocaliza-D2`), Transformer Encoder (64 dim, 4 cabezas) y motor normativo. | Diapositiva 5-6 / Código |
| **5:30 - 8:00** | **Demostración Funcional en Vivo** | Ejecución de la app: detección multiobjeto con cajas y clases, modo comparativo GT vs Predicción (IoU ≥ 0.5), y consulta al Asistente Transformer. | Aplicación Streamlit en Vivo |
| **8:00 - 9:00** | **Resultados y Evaluación** | Métricas en test reservado (Precision, Recall, F1, IoU emparejado), clases no evaluables y análisis de casos de error. | Diapositiva 7 |
| **9:00 - 10:00** | **Conclusiones y Cierre** | Cumplimiento estricto de la regla de no preentrenados, explicabilidad y balance entre visión y reglas. | Diapositiva 8 |

---

## 2. Guion Detallado de Exposición (Qué decir en cada minuto)

### Minuto 0:00 - 1:30: Apertura
> *"Buenos días, profesor Cristian y compañeros. Soy estudiante de Ingeniería de Software de la Universidad Cooperativa de Colombia, campus Pasto. Hoy presento la sustentación del Proyecto 4: Detección de residuos y clasificación de basura.*  
> *En Colombia, la Resolución 2184 de 2019 estableció un código unificado de colores para separar residuos: Blanco para aprovechables limpios y secos, Verde para orgánicos aprovechables y Negro para no aprovechables. Sin embargo, en la vida cotidiana las personas dudan sobre cómo clasificar objetos mixtos o sucios.*  
> *El reto fundamental de este proyecto consistió en construir una solución completa —detector visual multiobjeto y asistente inteligente— **entrenada 100% desde cero**, sin descargar checkpoints externos, sin backbones comerciales como YOLO preentrenado ni embeddings de lenguaje preentrenados."*

### Minuto 1:30 - 3:30: Datos y Partición
> *"Para cumplir con el rigor científico exigido, diseñamos nuestra propia herramienta de anotación interactiva en Python y OpenCV. Esta herramienta permite cargar grabaciones de video, navegar fotograma a fotograma y trazar con el mouse los recuadros delimitadores para las seis clases del proyecto: plástico, papel, cartón, vidrio, metal y restos orgánicos.*  
> *Un aspecto crítico de nuestra metodología fue la **partición estricta por sesión de captura**: todos los fotogramas de una misma grabación van exclusivamente a train, a val o a test. Esto previene el fenómeno de 'data leakage' o fuga de datos, donde fotogramas casi idénticos de un video inflarían falsamente las métricas.*  
> *Además, etiquetamos fotogramas 'revisados vacíos' como ejemplos negativos, enseñándole al detector a suprimir falsas alarmas sobre superficies limpias."*

### Minuto 3:30 - 5:30: Arquitectura
> *"En la parte visual implementamos `WasteDetectorNet`, un detector de disparo único en cuadrícula de 16x16, diseñado para ser compacto y explicable en un portátil. Consta de 5 bloques convolucionales con normalización por lotes (BatchNorm) y activación LeakyReLU, inicializados con la técnica Kaiming Normal.*  
> *La función de pérdida combina tres términos: una pérdida focal de presencia para resolver el desbalance de fondo, pérdida Smooth L1 para las coordenadas y dimensiones de la caja, y entropía cruzada para clasificar los materiales.*  
> *Para el asistente, implementamos un **Transformer Encoder** en PyTorch con embeddings posicionales y dos capas de auto-atención multi-cabeza, con vocabulario construido desde cero sobre consultas en español. El transformer clasifica la intención de la pregunta y el material mencionado, y se conecta con una tabla normativa que resuelve ambigüedades situacionales, por ejemplo, cuando el usuario pregunta '¿Dónde lo boto?' y la cámara detecta dos residuos diferentes."*

### Minuto 5:30 - 8:00: Demostración en Vivo
*(Abres la aplicación en el navegador)*:
1. **Detección Multiobjeto:** Subes una imagen o muestras a la cámara una botella de plástico y una lata de metal.
   > *"Como observan, el modelo localiza ambos objetos simultáneamente en la misma escena. Cada uno recibe su propia caja delimitadora, clase y porcentaje de confianza. A la derecha, el sistema clasifica el grupo operativo —ambos inorgánicos— y asigna la Caneca Blanca según la Resolución 2184."*
2. **Modo Comparativo:** Cambias a la pestaña de Modo Comparativo.
   > *"Aquí observamos el formato exigido en la guía: a la izquierda la predicción del modelo y a la derecha la anotación humana de referencia. El sistema calcula en tiempo real el IoU para cada objeto; al superar el umbral de 0.5 con emparejamiento uno a uno, se valida como detección correcta."*
3. **Asistente Transformer:** Cambias a la pestaña del Asistente y pulsas «¿Dónde lo boto?»:
   > *"Al haber dos residuos en pantalla, el Transformer reconoce la intención 'donde_botar_ambiguo' y el motor responde advirtiendo la ambigüedad: 'Veo varios residuos: Plástico y Metal. ¿A cuál te refieres?'. Si luego pregunto '¿Qué hago si el cartón está sucio?', el asistente aclara que la cámara detecta el material, pero si tiene grasa o comida, debe ir a la Caneca Negra."*

### Minuto 8:00 - 10:00: Métricas, Conclusiones y Preguntas
> *"En el conjunto de prueba reservado evaluamos precisión y exhaustividad con emparejamiento uno a uno bajo IoU ≥ 0.5. Como reportamos honestamente en el informe, las clases que no tuvieron muestras en test quedan documentadas explícitamente.*  
> *En conclusión, demostramos que es viable entrenar arquitecturas compactas de visión y transformers desde cero para resolver un problema ambiental concreto y normado en Colombia, manteniendo total control sobre los pesos y la explicabilidad del sistema. Quedo atento a sus preguntas."*

---

## 3. Banco de Preguntas de Sustentación con Respuestas Técnicas (Terminología del Profesor)

### Pregunta 1: ¿Qué es una convolución matemática y cómo extrae características de los residuos?
- **Respuesta:**
  > *"Una convolución en 2D es una operación matemática entre una matriz de entrada (la imagen o mapa de características anterior) y un filtro o kernel $K$ de tamaño $k \times k$. Consiste en desplazar el kernel sobre la imagen, multiplicando elemento a elemento y sumando los productos:  
  > $$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i-m, j-n) K(m,n)$$  
  > En nuestro código (`conv_block`), usamos kernels de $3 \times 3$. En las primeras capas (etapas 1 y 2), los filtros aprenden detectores de bajo nivel como bordes rectos, esquinas y gradientes de color (diferenciando el brillo de una lata metálica del contorno de una botella). En capas más profundas (etapas 4 y 5), la combinación jerárquica de estos mapas representa patrones complejos de forma y textura de los residuos."*

---

### Pregunta 2: ¿Cómo predice la red las cajas delimitadoras y cómo se decodifican los offsets?
- **Respuesta:**
  > *"Nuestra red divide la imagen en una cuadrícula de $S \times S$ celdas (16x16). Cada celda es responsable de predecir si el centro de un residuo cae dentro de ella.  
  > Para que las coordenadas sean estables durante el entrenamiento:  
  > 1. Aplicamos una función **sigmoide** $\sigma$ a las predicciones de desplazamiento $(t_x, t_y)$, restringiendo el centro dentro de la celda:  
  >    $$c_x = (j + \sigma(t_x)) \cdot \text{stride}, \quad c_y = (i + \sigma(t_y)) \cdot \text{stride}$$  
  > 2. Para el ancho y alto $(w, h)$, aplicamos **Softplus** ($f(x) = \ln(1 + e^x)$), garantizando que las dimensiones sean estrictamente positivas sin peligro de indeterminación matemática.  
  > 3. Finalmente, calculamos las esquinas $x_1 = c_x - w/2$ y $x_2 = c_x + w/2$."*

---

### Pregunta 3: ¿Qué es el IoU (Intersección sobre Unión) y por qué se exige emparejamiento uno a uno?
- **Respuesta:**
  > *"El IoU es la métrica estándar para medir la calidad de localización de una caja respecto a la verdad terreno (Ground Truth). Se define como el área de la intersección dividida entre el área de la unión:  
  > $$\text{IoU} = \frac{\text{Área}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Área}(B_{\text{pred}} \cup B_{\text{gt}})}$$  
  > Se exige un emparejamiento **uno a uno** (con umbral IoU $\ge 0.5$) para evitar evaluar con trampa: si un modelo genera 10 cajas sobre una misma botella, solo la caja con mayor confianza puede emparejarse como Verdadero Positivo (TP); las 9 restantes deben penalizarse como Falsos Positivos (FP). De lo contrario, un modelo redundante inflaría artificialmente su Recall."*

---

### Pregunta 4: ¿Qué función de pérdida se utilizó y cómo balancea las diferentes tareas?
- **Respuesta:**
  > *"Implementamos una función de pérdida multiobjetivo $\mathcal{L}_{\text{total}}$ en `scripts/train_waste_detector.py`:  
  > $$\mathcal{L}_{\text{total}} = \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$  
  > 1. $\mathcal{L}_{\text{obj}}$ es una entropía cruzada binaria con logits (BCEWithLogits) sobre todas las $16 \times 16 = 256$ celdas. Como la mayoría están vacías, aplicamos un sesgo inicial negativo $\ln(0.01/0.99)$ para evitar que la red colapse prediciendo objetos en todas partes.  
  > 2. $\mathcal{L}_{\text{box}}$ es una pérdida **Smooth L1** (Huber), calculada **únicamente sobre las celdas positivas** (donde hay un residuo real). Es cuadrática para errores pequeños y lineal para errores grandes, lo que la hace robusta frente a valores atípicos.  
  > 3. $\mathcal{L}_{\text{cls}}$ es una pérdida de **Cross-Entropy** estándar entre las 6 clases del proyecto, también evaluada solo en las celdas positivas."*

---

### Pregunta 5: ¿Cómo funciona la retropropagación (Backpropagation) y qué optimizador usamos?
- **Respuesta:**
  > *"La retropropagación aplica la **regla de la cadena del cálculo diferencial** para propagar el gradiente del error desde la función de pérdida hacia atrás a través de todas las capas de la red:  
  > $$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \cdot (A^{[l-1]})^T$$  
  > Usamos el optimizador **AdamW** con tasa de aprendizaje $\eta = 0.001$. AdamW calcula momentos de primer orden (promedio móvil del gradiente) y segundo orden (promedio móvil de los gradientes al cuadrado) para adaptar la tasa de aprendizaje por cada parámetro, y desacopla la penalización de decaimiento de pesos (Weight Decay) para evitar que los pesos crezcan desproporcionadamente."*

---

### Pregunta 6: ¿Cómo funciona el mecanismo de auto-atención en el Transformer y por qué se divide la dimensión entre las cabezas?
- **Respuesta:**
  > *"En nuestro `WasteAssistantTransformer`, cada palabra se proyecta linealmente en tres vectores: Consulta ($Q$, Query), Clave ($K$, Key) y Valor ($V$, Value). La atención del producto punto escalado calcula qué tanta relevancia tiene cada palabra respecto a las demás:  
  > $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$  
  > El factor $\sqrt{d_k}$ (donde $d_k = d_{\text{model}} / h = 64 / 4 = 16$) evita que los productos punto crezcan en magnitud y saturen la función softmax en regiones de gradiente casi nulo.  
  > Tener **4 cabezas de atención (Multi-Head)** permite a la red atender simultáneamente a diferentes tipos de relaciones sintácticas y semánticas: una cabeza puede asociar verbos con preposiciones ('separar' con 'de'), mientras otra asocia adjetivos con sustantivos ('cartón' con 'sucio')."*

---

### Pregunta 7: ¿Por qué está prohibido usar modelos preentrenados y cuál es el mérito de entrenar desde cero?
- **Respuesta:**
  > *"Un modelo preentrenado (como YOLOv8 o BERT descargado de la web) ya vio millones de imágenes de COCO o textos de Wikipedia. Usarlo oculta el aprendizaje fundamental de la ingeniería de software y la inteligencia artificial: el alumno no diseña la arquitectura, no experimenta la convergencia de gradientes, no sufre el desbalance de clases ni aprende a inicializar los pesos.  
  > Al inicializar con **Kaiming Normal** (que ajusta la varianza según la cantidad de conexiones de entrada $\text{Var}(W) = 2/n_{\text{in}}$ para evitar que los gradientes se desvanezcan o exploten en ReLU), garantizamos que cada parámetro fue aprendido única y exclusivamente con los datos de nuestro proyecto, demostrando dominio pleno del ciclo de vida de la IA."*
