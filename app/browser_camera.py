"""Browser camera frames sent through Streamlit's existing HTTPS/WebSocket channel."""
import base64
from collections import deque
from dataclasses import replace
from pathlib import Path
import time

import streamlit.components.v1 as components

from app.detector import ObjectTracker, box_iou, crop_box
from app.localization import Region
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
        self.zoom_candidate=None
        self.zoom_retry_at=0

    def _detect_live(self, frame, detector):
        detections=detector.detect(frame)
        if detections:
            self.zoom_candidate=None
            self.zoom_retry_at=0
            return detections
        if self.sequence<self.zoom_retry_at:
            return []

        height,width=frame.shape[:2]
        left,top=round(width*.14),round(height*.14)
        zoom=frame[top:height-top,left:width-left].copy()
        zoom_height,zoom_width=zoom.shape[:2]
        if min(zoom_height,zoom_width)<32:
            return []
        margin=max(2,min(width,height)*.012)
        candidate=None
        for detection in detector.detect(zoom):
            x1,y1,x2,y2=detection.box
            if min(x1,y1,zoom_width-x2,zoom_height-y2)<margin:
                continue
            box=(x1+left,y1+top,x2+left,y2+top)
            region=Region(crop_box(box,width,height),'neuronal-VisionLocaliza-D2')
            candidate=replace(detection,box=box,region=region)
            break

        if candidate is None:
            self.zoom_candidate=None
            self.zoom_retry_at=self.sequence+3
            return []
        now=time.monotonic()
        previous=self.zoom_candidate
        self.zoom_candidate=(candidate.box,now)
        same_candidate=previous and now-previous[1]<1.1 and box_iou(candidate.box,previous[0])>=.5
        active=next(iter(self.tracker.tracks.values()),None)
        same_object=active and now-self.tracker.last_seen<1.1 and box_iou(candidate.box,active['box'])>=.5
        return [candidate] if same_candidate or same_object else []

    def consume(self,payload,detector,classifier):
        if not isinstance(payload,dict):return False
        sid,sequence=payload.get('stream_id'),payload.get('sequence')
        if not isinstance(sid,str) or not 1<=len(sid)<=80 or type(sequence) is not int or sequence<1:return False
        if sid in self.retired:return False
        if sid==self.stream_id and sequence<=self.sequence:return False
        if sid!=self.stream_id:
            if self.stream_id:self.retired.append(self.stream_id)
            self.stream_id=sid;self.sequence=0;self.tracker=ObjectTracker(live=True);self.objects=()
            self.zoom_candidate=None;self.zoom_retry_at=0
        self.sequence=sequence
        self.response={'stream_id':sid,'sequence':sequence,'detection':None}
        if payload.get('event')=='stopped':
            self.running=False;self.objects=();self.tracker=ObjectTracker(live=True)
            self.zoom_candidate=None;self.zoom_retry_at=0
            return True
        try:
            if payload.get('event')!='frame':raise ValueError('Fotograma inválido.')
            encoded=payload.get('image')
            if not isinstance(encoded,str) or len(encoded)>2_000_000:raise ValueError('Fotograma demasiado grande.')
            frame=decode_image(base64.b64decode(encoded,validate=True),max_bytes=1_500_000,max_pixels=640*640)
            objects=self.tracker.update(frame,self._detect_live(frame,detector),classifier)
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
            self.zoom_candidate=None;self.zoom_retry_at=0
            self.response['error']='No se pudo reconocer este fotograma. Prueba con más luz y el residuo completo.'
        return True
