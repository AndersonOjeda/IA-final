"""Aplicación principal de Vision-IA (Proyecto 4: Detección y Clasificación de Residuos).
Universidad Cooperativa de Colombia - Campus Pasto.
Docente: Ing. Cristian Camilo Ordoñez Quintero.

Ejecutar con:
    streamlit run app/main.py
"""
import hashlib
import html
import os
import sys
from pathlib import Path

# Configurar para evitar problemas con libiomp5md en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

from app.detector import analyze_all_objects, analyze_with_waste_detector_net, annotate_objects
from app.localization import manual_region
from app.recycling import guidance, LABELS
from app.resources import (
    get_detector, get_model, get_assistant, get_waste_detector_net, reserved_image_hashes
)
from app.browser_camera import BrowserRecognition, CAMERA_GUIDE, camera_component
from app.settings import checkpoint_signature, load_settings
from app.vision import decode_image, annotate_region
from training.assistant_data import ID_TO_INTENT
from training.assistant_transformer import ID_TO_MAT
from utils.annotation_storage import AnnotationDataset, CLASS_DISPLAY, CLASSES
from utils.visual_comparison import create_side_by_side_comparison
from training.multi_waste_detector import calculate_iou


def live_panel(service, detector, threshold, model_key):
    previous = st.session_state.get('browser_recognition')
    if previous is None or previous[0] != model_key:
        previous = (model_key, BrowserRecognition(threshold))
        st.session_state['browser_recognition'] = previous
    recognition = previous[1]
    def receive_frame():
        recognition.consume(st.session_state.get('browser_camera'), detector, service)
    payload = camera_component(
        response=recognition.response,
        guide=CAMERA_GUIDE,
        interval_ms=250,
        key='browser_camera',
        on_change=receive_frame,
        default=None
    )
    recognition.consume(payload, detector, service)


def show_object_card(reading, threshold: float = 0.35):
    info = guidance(reading.material)
    uncertain = reading.confidence < threshold
    title = ('Posible ' if uncertain else '') + info['label'].lower()
    color = 'green' if reading.material == 'organicos' else 'white'
    
    st.markdown(
        f'<div class="waste-result">'
        f'<h3>{html.escape(title.capitalize())} <span style="font-size:0.8em; color:#888;">({reading.confidence*100:.1f}%)</span></h3>'
        f'<p class="destination"><span class="bin-dot {color}"></span>'
        f'<strong>Grupo:</strong> {html.escape(info["group"])} | <strong>Destino:</strong> {html.escape(info["destination"])}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.write(f"💡 **Recomendación:** {info['advice']}")
    for tip in info.get("tips", []):
        st.caption(f"• {tip}")
    if uncertain:
        st.warning('⚠️ Confianza moderada. Se sugiere mejorar iluminación o enfocar más de cerca.')


def run_assistant_query(query: str, scene_objects: list, tokenizer, model, engine) -> dict:
    if not tokenizer or not model or not engine:
        return {
            "respuesta": "El modelo Transformer del asistente no está cargado. Entrénalo ejecutando: python -m scripts.train_assistant_transformer",
            "intent": "desconocido",
            "material": "none"
        }
    
    device = next(model.parameters()).device
    ids, mask = tokenizer.encode(query)
    t_ids = torch.tensor([ids], device=device)
    t_mask = torch.tensor([mask], device=device)
    
    with torch.no_grad():
        i_logits, m_logits = model(t_ids, t_mask)
        p_intent = ID_TO_INTENT[i_logits.argmax().item()]
        p_mat = ID_TO_MAT[m_logits.argmax().item()]
        
    scene_data = []
    for item in scene_objects:
        mat = getattr(item.reading, "material", "desconocido")
        conf = getattr(item.reading, "confidence", 0.0)
        scene_data.append({"class_name": mat, "confidence": conf})
        
    res = engine.answer_query(p_intent, p_mat, scene_data)
    res["intent"] = p_intent
    res["material"] = p_mat
    return res


def main():
    st.set_page_config(
        page_title="Vision-IA | UCC Pasto",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    css_path = ROOT / "app/style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    # Encabezado institucional
    st.title("♻️ Vision-IA: Detección y Clasificación de Residuos")
    st.markdown(
        "**Universidad Cooperativa de Colombia — Campus Pasto** | Ingeniería de Software  \n"
        "*Proyecto 4 — Inteligencia Artificial (Prof. Cristian Camilo Ordoñez Quintero)*  \n"
        "Sistema de visión artificial y transformer entrenados **desde cero** bajo la **Resolución 2184 de 2019** (Código de Colores de Colombia)."
    )

    settings = load_settings()

    # Cargar modelos con manejo seguro de excepciones
    try:
        model_key = (*checkpoint_signature(settings['checkpoint']), settings['cpu_threads'])
        if settings.get('material_candidate_checkpoint'):
            model_key += (
                *checkpoint_signature(settings['material_candidate_checkpoint']),
                settings.get('material_candidate_threshold', 0.4)
            )
        service = get_model(*model_key)

        detector_key = (
            *checkpoint_signature(settings['detector_checkpoint']),
            settings['cpu_threads'],
            settings['detection_threshold'],
            settings.get('detection_nms_iou', 0.45)
        )
        detector = get_detector(*detector_key)
    except Exception as exc:
        st.error(
            f"❌ **Error al cargar los pesos del detector o clasificador visual:** {exc}  \n"
            "Asegúrate de haber generado los checkpoints propios con:  \n"
            "`python -m scripts.train_waste_detector`"
        )
        return

    assistant_data = get_assistant()
    if assistant_data is not None:
        ast_tokenizer, ast_model, ast_engine = assistant_data
    else:
        ast_tokenizer = ast_model = ast_engine = None

    waste_net = get_waste_detector_net()

    # Barra lateral con información técnica y parámetros
    with st.sidebar:
        st.header("⚙️ Panel de Control")
        st.caption("Ajuste de inferencia en tiempo real")
        
        engine_options = []
        if waste_net is not None:
            engine_options.append("⭐ WasteDetectorNet (Convoluciones Profundas - Plástico vs Vidrio)")
        engine_options.append("Pipeline Clásico en Cascada (D2 + CNN B)")
        selected_engine = st.selectbox("Motor de Inferencia Visual", engine_options, index=0)

        conf_default = 0.28 if "WasteDetectorNet" in selected_engine else float(settings['detection_threshold'])
        conf_slider = st.slider("Umbral de Confianza", 0.10, 0.90, conf_default, 0.05)
        use_smoothing = st.checkbox(
            "🌊 Suavizado Temporal (Anti-Parpadeo)",
            value=True,
            help="Acumula votos y suaviza coordenadas entre fotogramas consecutivos para estabilidad perfecta en cámara."
        )
        st.divider()
        st.markdown("### 🏷️ Clases del Proyecto (6)")
        for c in CLASSES:
            grp = "Orgánico" if c == "organicos" else "Inorgánico"
            st.markdown(f"- **{CLASS_DISPLAY[c]}** *({grp})*")
        st.divider()
        st.markdown("### 🇨🇴 Código de Colores (Res. 2184)")
        st.markdown("⚪ **Blanco:** Aprovechables limpios (plástico, papel, cartón, vidrio, metal).")
        st.markdown("🟢 **Verde:** Orgánicos aprovechables (restos de comida, podas).")
        st.markdown("⚫ **Negro:** No aprovechables (papel higiénico, servilletas, cartón grasoso).")

    # Pestañas principales de navegación
    tab_multi, tab_live, tab_compare, tab_assistant, tab_data = st.tabs([
        "📸 Detección Multiobjeto (Foto / Imagen)",
        "🎥 Cámara en Vivo",
        "⚖️ Modo Comparativo (GT vs Predicción)",
        "💬 Asistente Transformer (Español)",
        "📊 Gestión de Datos y Anotaciones"
    ])

    # -------------------------------------------------------------
    # PESTAÑA 1: Detección Multiobjeto (Imágenes / Fotogramas)
    # -------------------------------------------------------------
    with tab_multi:
        st.subheader("Localización y Clasificación de Múltiples Residuos en la Escena")
        st.caption("Permite identificar varios residuos simultáneos con sus recuadros, clases y porcentajes de confianza.")

        upload_col1, upload_col2 = st.columns([1, 1])
        with upload_col1:
            upload_mode = st.radio("Método de entrada", ["Subir archivo de imagen", "Tomar fotografía con cámara"], horizontal=True)
            if upload_mode == "Subir archivo de imagen":
                uploaded_file = st.file_uploader("Carga una fotografía con uno o varios residuos", type=["jpg", "jpeg", "png", "webp"])
            else:
                uploaded_file = st.camera_input("Capturar foto desde la webcam")

        if uploaded_file is not None:
            try:
                payload = uploaded_file.getvalue()
                rgb = decode_image(payload, settings['max_upload_mb'] * 1024 * 1024, settings['max_image_pixels'], reserved_image_hashes())
                
                with st.spinner("Localizando y clasificando residuos en la escena..."):
                    if "WasteDetectorNet" in selected_engine and waste_net is not None:
                        all_objects = analyze_with_waste_detector_net(rgb, waste_net, confidence=conf_slider, use_temporal_smoothing=use_smoothing)
                    else:
                        all_objects = analyze_all_objects(rgb, detector, service, confidence=conf_slider)

                # Guardar objetos en session_state para que el asistente pueda consultarlos
                st.session_state["current_scene_objects"] = all_objects

                col_view, col_details = st.columns([1.5, 1], gap="medium")
                with col_view:
                    annotated_img = annotate_objects(rgb, all_objects)
                    st.image(annotated_img, use_container_width=True, caption=f"Detección visual: {len(all_objects)} residuo(s) localizado(s)")

                with col_details:
                    st.markdown(f"### 📋 Residuos Detectados ({len(all_objects)})")
                    if not all_objects:
                        st.info("ℹ️ No se detectaron residuos con confianza suficiente. Prueba aumentando la iluminación o reduciendo el umbral de confianza.")
                    else:
                        for idx, item in enumerate(all_objects):
                            with st.expander(f"Residuo #{idx+1}: {CLASS_DISPLAY.get(item.reading.material, item.reading.material)}", expanded=True):
                                show_object_card(item.reading, threshold=conf_slider)

            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")

    # -------------------------------------------------------------
    # PESTAÑA 2: Cámara en Vivo
    # -------------------------------------------------------------
    with tab_live:
        st.subheader("Detección en Video en Tiempo Real")
        st.caption("Transmisión de video desde la cámara del navegador con seguimiento y filtrado temporal.")
        live_panel(service, detector, conf_slider, (model_key, detector_key))

    # -------------------------------------------------------------
    # PESTAÑA 3: Modo Comparativo (Estilo Imagen de Referencia)
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("⚖️ Modo Comparativo: Predicciones del Modelo vs Anotaciones Humanas")
        st.caption("Visualización lado a lado tal como se solicita en la guía del proyecto, evaluando coincidencia IoU ≥ 0.5.")

        dataset_annot = AnnotationDataset()
        records_with_boxes = [r for r in dataset_annot.records.values() if len(r.boxes) > 0 and Path(r.image_path).exists()]

        if not records_with_boxes:
            st.info(
                "ℹ️ Aún no hay fotogramas anotados con recuadros reales en `data/annotations/annotations.json`.  \n"
                "Para generar anotaciones de prueba, ejecuta la herramienta interactiva:  \n"
                "`python -m scripts.annotate_tool --source ruta/a/video.mp4`"
            )
        else:
            options = [f"{r.video_id} - Fotograma #{r.frame_index} ({r.split})" for r in records_with_boxes]
            chosen_label = st.selectbox("Selecciona un fotograma de prueba anotado:", options)
            chosen_rec = records_with_boxes[options.index(chosen_label)]

            img_bgr = cv2.imread(chosen_rec.image_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                preds = analyze_all_objects(img_rgb, detector, service, confidence=conf_slider)
                
                # Transformar predicciones a formato compatible con comparador
                pred_dicts = []
                for p in preds:
                    pred_dicts.append({
                        "box": p.detection.box,
                        "class_name": p.reading.material,
                        "confidence": p.reading.confidence
                    })

                gt_dicts = chosen_rec.boxes

                comparison_img = create_side_by_side_comparison(
                    img_rgb, pred_dicts, gt_dicts, title_suffix=f"[{chosen_rec.split.upper()}]"
                )
                st.image(comparison_img, use_container_width=True)

                # Tabla de cálculo de IoU para emparejamientos
                st.markdown("#### 📐 Cálculo de Coincidencia (IoU) por Objeto:")
                comp_cols = st.columns(len(gt_dicts) if gt_dicts else 1)
                for i, gt in enumerate(gt_dicts):
                    best_iou = 0.0
                    matched_p = None
                    for p in pred_dicts:
                        if p["class_name"] == gt["label"]:
                            iou_val = calculate_iou(p["box"], (gt["x1"], gt["y1"], gt["x2"], gt["y2"]))
                            if iou_val > best_iou:
                                best_iou = iou_val
                                matched_p = p
                    
                    status = "✅ Coincidencia Válida (IoU ≥ 0.5)" if best_iou >= 0.5 else "❌ Sin Coincidencia / IoU < 0.5"
                    st.write(f"**Objeto Real #{i+1} ({CLASS_DISPLAY[gt['label']]}):**")
                    st.write(f"- IoU obtenido: `{best_iou:.3f}`")
                    st.write(f"- Estado: {status}")

    # -------------------------------------------------------------
    # PESTAÑA 4: Asistente Transformer (Español)
    # -------------------------------------------------------------
    with tab_assistant:
        st.subheader("💬 Asistente Experto en Separación de Residuos (Transformer Propio)")
        st.markdown(
            "Este asistente utiliza una red **Transformer Encoder entrenada desde cero** en español "
            "para reconocer tu intención y entities, conectándola con los residuos detectados en cámara y la normativa colombiana."
        )

        # Contexto visual actual
        active_objects = st.session_state.get("current_scene_objects", [])
        if active_objects:
            detected_names = [CLASS_DISPLAY.get(o.reading.material, o.reading.material) for o in active_objects]
            st.success(f"👁️ **Objetos en la escena visual actual:** {', '.join(detected_names)}")
        else:
            st.info("👁️ **Cámara / Escena:** No hay objetos cargados actualmente. (Puedes probar consultas o subir una imagen en la pestaña 1).")

        # Botones de consulta rápida
        st.markdown("**Consultas sugeridas de ejemplo:**")
        q_cols = st.columns(3)
        sample_query = None
        with q_cols[0]:
            if st.button("«¿Qué residuos ves?»"):
                sample_query = "¿Qué residuos ves?"
            if st.button("«¿Cómo separo esta lata?»"):
                sample_query = "¿Cómo separo esta lata?"
        with q_cols[1]:
            if st.button("«¿Hay algo orgánico?»"):
                sample_query = "¿Hay algo orgánico?"
            if st.button("«¿Qué hago si el cartón está sucio?»"):
                sample_query = "¿Qué hago si el cartón está sucio?"
        with q_cols[2]:
            if st.button("«¿Dónde lo boto?» (Prueba de ambigüedad)"):
                sample_query = "¿Dónde lo boto?"
            if st.button("«¿Cuál es el clima de hoy?» (Fuera de alcance)"):
                sample_query = "¿Cuál es el clima de hoy?"

        user_input = st.text_input("Escribe tu pregunta sobre los residuos en español:", value=sample_query or "")

        if st.button("Consultar Asistente", type="primary") and user_input:
            with st.spinner("Procesando con el Transformer..."):
                answer = run_assistant_query(user_input, active_objects, ast_tokenizer, ast_model, ast_engine)

            st.markdown("### 🤖 Respuesta del Asistente:")
            st.markdown(f"> {answer['respuesta']}")

            with st.expander("🔍 Detalles Técnicos del Transformer (Explicabilidad)", expanded=False):
                st.write(f"• **Intención clasificada por el Transformer:** `{answer.get('intent')}`")
                st.write(f"• **Material identificado en la consulta:** `{answer.get('material')}`")
                st.write("• **División de responsabilidades:**")
                st.caption(
                    "- **Transformer (Aprendido desde cero):** Comprensión del lenguaje natural, extracción de intención y detección del material en la frase.  \n"
                    "- **Motor Normativo (Reglas Colombia):** Vinculación con los objetos reales de la cámara, manejo de ambigüedad si hay múltiples residuos y asignación del color de caneca según Res. 2184."
                )

    # -------------------------------------------------------------
    # PESTAÑA 5: Gestión de Datos y Anotaciones
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("📊 Resumen del Dataset y Anotaciones")
        st.markdown(
            "Comando de consola para auditar particiones y objetos:  \n"
            "`python -m scripts.data_summary`"
        )
        db_inst = AnnotationDataset()
        stats = db_inst.summary()
        
        stat_cols = st.columns(4)
        stat_cols[0].metric("Videos Registrados", stats["total_videos"])
        stat_cols[1].metric("Sesiones de Captura", stats["total_sessions"])
        stat_cols[2].metric("Fotogramas Totales", stats["total_images"])
        stat_cols[3].metric("Vacíos Revisados", sum(stats["split_reviewed_empty"].values()))

        st.markdown("#### Objetos Anotados por Clase y Partición:")
        rows = []
        for cls in CLASSES:
            c_train = stats["objects_per_class"]["train"][cls]
            c_val = stats["objects_per_class"]["val"][cls]
            c_test = stats["objects_per_class"]["test"][cls]
            rows.append({
                "Clase": CLASS_DISPLAY[cls],
                "Grupo": "Orgánico" if cls == "organicos" else "Inorgánico",
                "Train": c_train,
                "Val": c_val,
                "Test": c_test,
                "Total": c_train + c_val + c_test
            })
        st.table(rows)


if __name__ == "__main__":
    main()
