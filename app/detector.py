"""Localizador VisionLocaliza-D2, NMS y recortes para las CNN existentes."""
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch

from app.localization import Region, square_box
from training.detector_own import ARCHITECTURE, VisionLocaliza, decode_predictions
from training.detection_data import letterbox, image_tensor
from training.detection_metrics import box_iou, nms


@dataclass(frozen=True)
class Detection:
    box: tuple
    confidence: float
    region: Region
    inference_ms: float = 0.


def crop_box(box,width,height,margin=.12):
    x1,y1,x2,y2=box
    dx,dy=max((x2-x1)*margin,(16-(x2-x1))/2),max((y2-y1)*margin,(16-(y2-y1))/2)
    expanded=(max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),
              min(width,int(np.ceil(x2+dx))),min(height,int(np.ceil(y2+dy))))
    if expanded[2]-expanded[0]<16 or expanded[3]-expanded[1]<16:
        left=min(max(0,expanded[0]),width-16)
        top=min(max(0,expanded[1]),height-16)
        expanded=(left,top,min(width,max(left+16,expanded[2])),min(height,max(top+16,expanded[3])))
    return square_box(expanded,width,height)


class NeuralDetector:
    def __init__(self,checkpoint,cpu_threads=2,confidence=.35,nms_iou=.45):
        path=Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError('Falta el detector propio D2 entrenado. Puedes usar el modo manual.')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        from utils.dataset import sha256
        self.checkpoint_sha256=sha256(path)
        provenance=payload.get('provenance',{})
        if provenance.get('pretrained') is not False or provenance.get('initialization')!='random':
            raise ValueError('Detector sin procedencia de entrenamiento desde cero.')
        if payload.get('architecture')!=ARCHITECTURE or provenance.get('official_detector_components') is not False:
            raise ValueError('Se exige arquitectura propia VisionLocaliza-D2. D1/YOLO no es válido como detector final.')
        torch.set_num_threads(cpu_threads)
        self.cpu_threads=cpu_threads
        self.model=VisionLocaliza()
        self.model.load_state_dict(payload['state_dict'],strict=True)
        self.model.eval().float()
        self.parameter_count=sum(p.numel() for p in self.model.parameters())
        self.size=int(payload['image_size'])
        self.confidence,self.nms_iou=confidence,nms_iou
        self.provenance=provenance
        self._lock=threading.Lock()

    def detect(self,rgb,confidence=None):
        # Streamlit may invoke a cached model from a new script/callback thread.
        threads=getattr(self,'cpu_threads',2)
        if torch.get_num_threads()!=threads:torch.set_num_threads(threads)
        if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Se requiere imagen RGB uint8.')
        started=time.perf_counter()
        height,width=rgb.shape[:2]
        canvas,_,(scale_x,scale_y,left,top)=letterbox(rgb,self.size)
        tensor=image_tensor(canvas).unsqueeze(0)
        with self._lock,torch.inference_mode():
            raw=self.model(tensor)
            decoded=decode_predictions(raw,self.confidence if confidence is None else confidence)[0]
            boxes,scores=decoded[0].cpu().numpy(),decoded[1].cpu().numpy()
        if len(scores)==0:
            return []
        boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/scale_x
        boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/scale_y
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
        valid=np.isfinite(boxes).all(axis=1)&np.isfinite(scores)&((boxes[:,2]-boxes[:,0])>=2)&((boxes[:,3]-boxes[:,1])>=2)
        boxes,scores=boxes[valid],scores[valid]
        elapsed=(time.perf_counter()-started)*1000
        # Hasta 100 para evaluar AP estándar; la aplicación limita solo el número de CNN.
        return [Detection(tuple(map(float,boxes[i])),float(scores[i]),
                    Region(crop_box(boxes[i],width,height),'neuronal-VisionLocaliza-D2'),elapsed)
                for i in nms(boxes,scores,self.nms_iou,max_detections=100)]


@dataclass(frozen=True)
class ObjectReading:
    detection: Detection
    reading: object
    track_id: int = 0
    confirmed: bool = True


def primary_detection(detections,previous_box=None):
    """One object; retain a nearby candidate when its score is close to the best."""
    if not detections:return None
    best=max(detections,key=lambda d:d.confidence)
    if previous_box is not None:
        nearby=[d for d in detections if box_iou(d.box,previous_box)>=.25]
        if nearby:
            current=max(nearby,key=lambda d:box_iou(d.box,previous_box))
            if current.confidence>=best.confidence-.10:return current
    return best


class ObjectTracker:
    """Un residuo principal estable; elimina pérdidas sin cajas fantasma."""
    def __init__(self, *, live=False):
        self.live=live
        self.tracks={}
        self.next_id=1
        self.misses=0
        self.last_seen=0.

    def update(self,frame,detections,classifier):
        from app.stability import LiveMaterialFilter, PredictionFilter
        from dataclasses import replace
        result,new={},{}
        used=set()
        now=time.monotonic()
        if self.live and now-self.last_seen>1.1:self.tracks.clear()
        previous=next(iter(self.tracks.values()),None)
        selected=primary_detection(detections,previous['box'] if previous else None)
        if self.live and not selected:
            self.misses+=1
            if self.misses>=2:self.tracks.clear()
            return []
        self.misses=0
        self.last_seen=now
        for detection in ([selected] if selected is not None else []):
            candidates=[(box_iou(detection.box,t['box']),key) for key,t in self.tracks.items() if key not in used]
            match=max(candidates,default=(0,None))
            key=match[1] if match[0]>=.25 else self.next_id
            if match[0]<.25:
                self.next_id+=1
            used.add(key)
            track=self.tracks.get(key)
            if track is None:
                track={'box':detection.box,'filter':LiveMaterialFilter() if self.live else PredictionFilter()}
            box=detection.box if self.live else tuple(.25*a+.75*b for a,b in zip(track['box'],detection.box))
            detection=replace(detection,box=box)
            reading=classifier.classify(frame,region=detection.region)
            temporal=track['filter'].update(frame,reading)
            result[key]=ObjectReading(detection,temporal or reading,key,temporal is not None)
            new[key]={'box':box,'filter':track['filter']}
        self.tracks=new
        return list(result.values())


def analyze_objects(frame,detector,classifier):
    selected=primary_detection(detector.detect(frame))
    return [ObjectReading(selected,classifier.classify(frame,region=selected.region))] if selected else []


def analyze_all_objects(frame, detector, classifier, confidence=None, max_objects=15):
    """Localiza y clasifica múltiples residuos simultáneamente en el mismo fotograma."""
    detections = detector.detect(frame, confidence=confidence)
    results = []
    for d in detections[:max_objects]:
        try:
            reading = classifier.classify(frame, region=d.region)
            results.append(ObjectReading(d, reading))
        except Exception:
            continue
    return results


def annotate_objects(frame, objects):
    from app.recycling import LABELS
    import unicodedata
    output = frame.copy()
    
    # Paleta de colores BGR por clase
    colors = {
        "plastico": (255, 140, 0),    # Celeste / azul
        "papel": (240, 240, 240),      # Blanco
        "carton": (42, 110, 180),      # Naranja / marrón
        "vidrio": (220, 220, 0),       # Cyan / turquesa
        "metal": (180, 180, 180),      # Gris metálico
        "organicos": (34, 139, 34)     # Verde
    }

    for item in objects:
        x1, y1, x2, y2 = map(round, item.detection.box)
        mat = getattr(item.reading, "material", "desconocido")
        conf = getattr(item.reading, "confidence", item.detection.confidence)
        color = colors.get(mat, (21, 126, 117))
        
        # Recuadro principal
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Etiqueta con clase y porcentaje de confianza
        display_label = LABELS.get(mat, mat.capitalize())
        group_badge = "Orgánico" if mat == "organicos" else "Inorgánico"
        text = f"{display_label} ({conf * 100:.1f}%)"
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
        
        # Fondo de la etiqueta para alta legibilidad
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        tag_y = max(th + 6, y1)
        cv2.rectangle(output, (x1, tag_y - th - 5), (x1 + tw + 6, tag_y + 2), (25, 25, 25), -1)
        cv2.putText(output, text, (x1 + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        
    return output
