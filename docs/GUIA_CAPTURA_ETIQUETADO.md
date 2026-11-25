# Guía de Captura y Etiquetado de Residuos (Vision-IA UCC Pasto)

**Proyecto 4:** Detección de residuos y clasificación de basura  
**Institución:** Universidad Cooperativa de Colombia — Campus Pasto  
**Docente:** Ing. Cristian Camilo Ordoñez Quintero  

---

## 1. Objetivo del Protocolo

Garantizar la recolección de datos realistas, variados y libres de fuga de información (*data leakage*) para el entrenamiento y evaluación del detector visual y el asistente inteligente, sin utilizar modelos preentrenados.

---

## 2. Clases Oficiales y Convención de Separación

El sistema distingue **6 clases de residuos** organizadas bajo la **Resolución 2184 de 2019** del Ministerio de Ambiente y Desarrollo Sostenible de Colombia:

| ID | Clase | Grupo Operativo | Destino Colombia | Ejemplos Típicos |
|---|---|---|---|---|
| 1 | `plastico` | Inorgánico | Caneca / Bolsa Blanca | Botellas PET, vasos descartables, tapas, bolsas plásticas limpias |
| 2 | `papel` | Inorgánico | Caneca / Bolsa Blanca | Hojas de cuaderno, impresos, sobres limpios y secos |
| 3 | `carton` | Inorgánico | Caneca / Bolsa Blanca | Cajas de embalaje, tubos de cartón, empaques secos y doblados |
| 4 | `vidrio` | Inorgánico | Caneca / Bolsa Blanca | Botellas de bebidas, frascos conserveros enteros |
| 5 | `metal` | Inorgánico | Caneca / Bolsa Blanca | Latas de aluminio (gaseosa/cerveza), latas de conserva, tapas metálicas |
| 6 | `organicos` | Orgánico | Caneca / Bolsa Verde | Cáscaras de fruta (plátano, naranja), restos de manzana, café, vegetales |

> [!IMPORTANT]
> **Convención Operativa:** En el contexto doméstico y normativo colombiano, papel y cartón se agrupan funcionalmente con los inorgánicos aprovechables (caneca blanca) siempre que estén **limpios y secos**. Si están contaminados con grasa o restos orgánicos, se deben depositar en la **caneca negra (no aprovechables)**.

---

## 3. Protocolo de Captura en Video

1. **Grabación por Sesiones Independientes:**
   - Graba videos cortos (15 a 45 segundos) en formato MP4 a resolución 720p o 1080p.
   - Cada grabación corresponde a una **sesión única** (`session_01`, `session_02`, etc.).
   - Varía el fondo (mesas de madera, mesones blancos, pisos, manteles) y la iluminación (luz natural de ventana, luz cálida de tungsteno, luz LED blanca).

2. **Escenas con Múltiples Residuos:**
   - Graba escenas donde aparezcan **2, 3 o más residuos simultáneos**, tanto de la misma clase (dos botellas de plástico) como de clases diferentes (una lata de metal junto a una cáscara de banano).
   - Realiza desplazamientos suaves de la cámara (paneo, acercamiento, inclinación).

3. **Fotogramas de Fondo Vacío (Ejemplos Negativos):**
   - Dedica al menos 3 a 5 segundos de cada video a enfocar la superficie de apoyo **sin ningún residuo**.
   - Estos fotogramas se marcan como **"Revisados Vacíos"** y enseñan a la red a no producir falsas detecciones (*false positives*).

---

## 4. Partición Estricta por Sesión (Anti-Leakage)

Para evitar sobrestimar la precisión del modelo:
- **Regla de Oro:** Todos los fotogramas provenientes de una misma sesión o video van al **mismo conjunto** (o todo a `train`, o todo a `val`, o todo a `test`).
- **Proporción Recomendada:**
  - 70% de sesiones a `train`
  - 15% de sesiones a `val` (para ajuste de hiperparámetros y umbrales)
  - 15% de sesiones a `test` (completamente reservadas para métricas finales)

---

## 5. Instrucciones de Uso de la Herramienta de Anotación

Ejecuta en PowerShell:

```powershell
python -m scripts.annotate_tool --source ruta/a/video.mp4 --session sesion_01
```

### Controles Interactivos:
- **Teclas 1 a 6:** Selecciona la clase activa (se visualiza el nombre y color en la barra superior).
- **Clic y Arrastre del Mouse:** Traza el recuadro ajustado alrededor del residuo. Procura que la caja contenga el objeto completo sin dejar excesivo margen de fondo.
- **Tecla `d`:** Elimina el último recuadro dibujado (útil si hubo error de pulso).
- **Tecla `c`:** Limpia todas las cajas del fotograma actual.
- **Tecla `e`:** Marca el fotograma como **"Revisado Vacío"** si no hay residuos presentes.
- **Espacio o `n`:** Avanza al siguiente fotograma (guarda automáticamente).
- **`p` o Backspace:** Retrocede al fotograma anterior.
- **`s`:** Guarda forzadamente las anotaciones en el archivo central `data/annotations/annotations.json`.
- **`q` o ESC:** Guarda y sale de la herramienta.

---

## 6. Verificación y Auditoría de Datos

Para conocer el estado actual de las grabaciones, fotogramas y objetos etiquetados, ejecuta:

```powershell
python -m scripts.data_summary
```

Este comando verifica:
- Cantidad de imágenes pendientes de etiquetar.
- Cantidad de imágenes revisadas vacías.
- Conteo de objetos por clase y por conjunto (`train`, `val`, `test`).
- Detección de clases que carezcan de muestras en el conjunto `test`.
