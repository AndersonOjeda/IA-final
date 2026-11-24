"""Browser camera frames sent through Streamlit's existing HTTPS/WebSocket channel."""
import base64
from collections import deque
from pathlib import Path

import streamlit.components.v1 as components

from app.detector import ObjectTracker
from app.recycling import guidance
from app.vision import decode_image

camera_component=components.declare_component('vision_camera',path=str(Path(__file__).parent/'camera_component'))


class BrowserRecognition:
    """Session-local tracking. No camera hardware, disk writes or remote models."""
    def __init__(self, material_threshold=.65):
        self.tracker=ObjectTracker(live=True)
        self.material_threshold=material_threshold
        self.stream_id=None
        self.sequence=0
        self.retired=deque(maxlen=8)
        self.objects=()
        self.response={}
        self.running=False

    def consume(self,payload,detector,classifier):
        if not isinstance(payload,dict):return False
        sid,sequence=payload.get('stream_id'),payload.get('sequence')
        if not isinstance(sid,str) or not 1<=len(sid)<=80 or type(sequence) is not int or sequence<1:return False
        if sid in self.retired:return False
        if sid==self.stream_id and sequence<=self.sequence:return False
        if sid!=self.stream_id:
            if self.stream_id:self.retired.append(self.stream_id)
            self.stream_id=sid;self.sequence=0;self.tracker=ObjectTracker(live=True);self.objects=()
        self.sequence=sequence
        self.response={'stream_id':sid,'sequence':sequence,'detection':None}
        if payload.get('event')=='stopped':
            self.running=False;self.objects=();self.tracker=ObjectTracker(live=True)
            return True
        try:
            if payload.get('event')!='frame':raise ValueError('Fotograma inválido.')
            encoded=payload.get('image')
            if not isinstance(encoded,str) or len(encoded)>2_000_000:raise ValueError('Fotograma demasiado grande.')
            frame=decode_image(base64.b64decode(encoded,validate=True),max_bytes=1_500_000,max_pixels=640*640)
            objects=self.tracker.update(frame,detector.detect(frame),classifier)
            self.objects=tuple(objects);self.running=True
            if objects:
                item=objects[0];height,width=frame.shape[:2]
                box=[float(v)/scale for v,scale in zip(item.detection.box,[width,height,width,height])]
                result=None
                if item.confirmed:
                    reading=item.reading
                    info=guidance(reading.material)
                    label=info['label']
                    if reading.confidence<self.material_threshold:label='Posible '+label.lower()
                    result={'label':label,'destination':info['destination'],
                            'organic':reading.material=='organicos'}
                self.response['detection']={'box':box,'track_id':item.track_id,'material':result}
        except Exception:
            self.objects=();self.tracker=ObjectTracker(live=True)
            self.response['error']='No se pudo reconocer este fotograma. Prueba con más luz y el residuo completo.'
        return True
