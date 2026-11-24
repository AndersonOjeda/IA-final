"""Cámara, material y destino. Ejecutar con streamlit run app/main.py."""
import hashlib
import html
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import streamlit as st

from app.detector import analyze_objects,annotate_objects
from app.localization import manual_region
from app.recycling import guidance
from app.resources import get_camera_pool,get_preview_server,get_detector,get_model,reserved_image_hashes
from app.settings import checkpoint_signature,load_settings
from app.vision import decode_image,annotate_region


def show_result(reading,threshold):
    info=guidance(reading.material)
    uncertain=reading.confidence<threshold
    title=('Posible ' if uncertain else '')+info['label'].lower()
    color='green' if reading.material=='organicos' else 'white'
    st.markdown(f'<div class="waste-result"><h3>{html.escape(title.capitalize())}</h3>'
                f'<p class="destination"><span class="bin-dot {color}"></span>'
                f'{html.escape(info["destination"])}</p></div>',unsafe_allow_html=True)
    st.caption(info['advice'])
    if uncertain:st.caption('No está claro el material. Acércalo y mejora la luz.')


def show_objects(objects,threshold):
    if not objects:st.info('Muestra un residuo completo frente a la cámara.')
    readings={}
    for item in objects:
        reading=item.reading
        previous=readings.get(reading.material)
        if previous is None or reading.confidence>previous.confidence:readings[reading.material]=reading
    for reading in readings.values():show_result(reading,threshold)


@st.fragment(run_every=.3)
def live_result(owner,threshold):
    stream=get_camera_pool().get(owner)
    snapshot=stream.snapshot() if stream else None
    if snapshot and snapshot.error:st.error(snapshot.error)
    elif snapshot and snapshot.frame is not None:
        fresh=time.monotonic()-snapshot.reading_at<1.
        if snapshot.analysis_frame is None:st.info('Buscando residuos…')
        else:show_objects(snapshot.objects if fresh else (),threshold)


def live_panel(owner,threshold):
    stream=get_camera_pool().get(owner)
    picture,result=st.columns([1.8,1],gap='large')
    with picture:
        if stream and stream.active:
            url=get_preview_server().url(stream)
            st.markdown(f'<img class="live-camera" alt="Cámara en vivo" src="{url}" />',unsafe_allow_html=True)
        else:st.markdown('<div class="camera-placeholder">Activa la cámara para reconocer un residuo.</div>',unsafe_allow_html=True)
    with result:live_result(owner,threshold)


def photo_panel(mode,service,detector,settings,model_key):
    upload=(st.camera_input('Tomar una fotografía',key='camera_photo') if mode=='Tomar foto' else
            st.file_uploader('Selecciona una imagen',type=['jpg','jpeg','png','webp'],key='upload_photo'))
    if upload is None:return
    try:
        payload=upload.getvalue()
        rgb=decode_image(payload,settings['max_upload_mb']*1024*1024,settings['max_image_pixels'],reserved_image_hashes())
        key=(hashlib.sha256(payload).hexdigest(),model_key)
        previous=st.session_state.get('photo_result')
        if previous is None or previous[0]!=key:
            with st.spinner('Reconociendo residuo…'):objects=analyze_objects(rgb,detector,service)
            st.session_state['photo_result']=(key,objects)
        else:objects=previous[1]
        picture,result=st.columns([1.8,1],gap='large')
        with picture:st.image(annotate_objects(rgb,objects),width='stretch')
        with result:
            if objects:show_objects(objects,settings['low_confidence_threshold'])
            else:st.info('No se reconoció un residuo. Prueba otra foto con más luz y el objeto completo.')
        with st.expander('Ajustar el recorte si no lo reconoce'):
            horizontal=st.slider('Horizontal (%)',0,100,(20,80),key='manual_x')
            vertical=st.slider('Vertical (%)',0,100,(20,80),key='manual_y')
            bounds=(horizontal[0]/100,vertical[0]/100,horizontal[1]/100,vertical[1]/100)
            region=manual_region(rgb,bounds)
            st.image(annotate_region(rgb,region),width=400)
            if st.button('Reconocer recorte'):
                show_result(service.classify(rgb,region=region),settings['low_confidence_threshold'])
    except Exception as exc:st.error(f'No se pudo reconocer la imagen: {exc}')


def main():
    st.set_page_config(page_title='Vision-IA',page_icon='♻',layout='wide',initial_sidebar_state='collapsed')
    st.markdown(f'<style>{(ROOT/"app/style.css").read_text(encoding="utf-8")}</style>',unsafe_allow_html=True)
    st.title('Vision-IA')
    st.caption('Muestra un residuo a la vez. Identifica el material y dónde va.')
    settings=load_settings()
    owner=st.session_state.setdefault('camera_owner',uuid.uuid4().hex)
    pool=get_camera_pool()
    mode=st.radio('Origen de la imagen',['Cámara en vivo','Tomar foto','Subir imagen'],horizontal=True,key='image_source',label_visibility='collapsed')
    try:
        model_key=(*checkpoint_signature(settings['checkpoint']),settings['cpu_threads'])
        service=get_model(*model_key)
        detector_key=(*checkpoint_signature(settings['detector_checkpoint']),settings['cpu_threads'],
                      settings['detection_threshold'],settings.get('detection_nms_iou',.45))
        detector=get_detector(*detector_key)
    except Exception as exc:
        pool.stop(owner);st.error(f'No se pudo cargar el reconocimiento: {exc}');return
    signature=(mode,model_key,detector_key)
    if st.session_state.get('capture_options')!=signature:
        pool.stop(owner);st.session_state['capture_options']=signature
    if mode=='Cámara en vivo':
        stream=pool.get(owner)
        active=stream is not None and stream.active
        start,stop,_=st.columns([1,1,4])
        if start.button('Activar cámara',type='primary',disabled=active,width='stretch'):
            try:
                pool.start(owner,service,index=settings['camera_index'],mode='neural',detector=detector,
                           interval=settings['inference_interval_seconds'])
                st.rerun()
            except Exception as exc:st.error(str(exc))
        if stop.button('Detener',disabled=not active,width='stretch'):
            pool.stop(owner);st.rerun()
        live_panel(owner,settings['low_confidence_threshold'])
    else:
        pool.stop(owner)
        photo_panel(mode,service,detector,settings,(model_key,detector_key))


if __name__=='__main__':main()
