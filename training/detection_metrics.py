"""Métricas espaciales y NMS; conceptos generales, sin modelos de terceros."""
from collections import Counter
import numpy as np


def box_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0.


def nms(boxes,scores,threshold=.45,max_detections=20):
    boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,4)
    scores=np.asarray(scores)
    order=np.argsort(-scores,kind='stable')
    areas=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    keep=[]
    while len(order) and len(keep)<max_detections:
        i=int(order[0]);keep.append(i)
        rest=order[1:]
        intersection=np.maximum(0,np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2])).prod(axis=1)
        overlap=intersection/np.maximum(areas[i]+areas[rest]-intersection,1e-9)
        order=rest[overlap<=threshold]
    return keep


def match(predictions,truth,threshold=.5):
    remaining=set(range(len(truth)))
    matches=[]
    for index,p in sorted(enumerate(predictions),key=lambda pair:-pair[1]['score']):
        best=max(((box_iou(p['box'],truth[j]['box']),j) for j in remaining),default=(0,None))
        if best[0]>=threshold:
            remaining.remove(best[1]);matches.append((index,best[1],best[0]))
    return matches


def spatial_metrics(records,confidence=.35):
    total=sum(len(r['truth']) for r in records)
    aps=[]
    for threshold in np.linspace(.5,.95,10):
        scored=[]
        for r in records:
            matched={i for i,j,iou in match(r['predictions'],r['truth'],threshold)}
            scored.extend((p['score'],int(i in matched)) for i,p in enumerate(r['predictions']))
        scored.sort(key=lambda p:-p[0])
        tp=np.cumsum([p[1] for p in scored])
        fp=np.cumsum([1-p[1] for p in scored])
        recall=tp/max(total,1)
        precision=tp/np.maximum(tp+fp,1)
        envelope=np.maximum.accumulate(precision[::-1])[::-1]
        aps.append(float(np.mean([max(envelope[recall>=r],default=0) for r in np.linspace(0,1,101)])))
    true_positives,predicted,ious=0,0,[]
    off_center_total,off_center_matched=0,0
    category_total,category_matched=Counter(),Counter()
    for r in records:
        ps=[p for p in r['predictions'] if p['score']>=confidence]
        matches=match(ps,r['truth'])
        true_positives+=len(matches);predicted+=len(ps)
        ious.extend(iou for i,j,iou in matches)
        category_total.update(t['category'] for t in r['truth'])
        category_matched.update(r['truth'][j]['category'] for i,j,iou in matches)
        if 'size' in r:
            w,h=r['size']
            off={i for i,t in enumerate(r['truth']) if not (.25<=(t['box'][0]+t['box'][2])/2/w<=.75 and .25<=(t['box'][1]+t['box'][3])/2/h<=.75)}
            off_center_total+=len(off)
            off_center_matched+=sum(j in off for i,j,iou in matches)
    return {'mAP50':aps[0],'mAP50_95':float(np.mean(aps)),'AP50_by_detector_class':{'residuo':aps[0]},
            'precision':true_positives/max(predicted,1),'recall':true_positives/max(total,1),
            'mean_IoU_matched_at_50':float(np.mean(ious)) if ious else 0.,'confidence_threshold':confidence,
            'images':len(records),'objects':total,'predicted_objects':predicted,'true_positives':true_positives,
            'source_category_recall':{c:{'recall':category_matched[c]/n,'objects':n} for c,n in category_total.items()},
            'off_center_objects':off_center_total,'off_center_recall':off_center_matched/max(off_center_total,1),
            'AP_method':'101-point interpolated AP, IoU 0.50:0.95 step .05, max 100 detections/image; one class',
            'category_note':'Source categories are diagnostics of localization recall, not learned material AP.'}


def coco_reference(records):
    """Contrasta el AP de una clase con la implementación de referencia COCO."""
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO()
    images=[];annotations=[];predictions=[]
    for r in records:
        images.append({'id':r['image_id']})
        for t in r['truth']:
            x,y,x2,y2=t['box'];w,h=x2-x,y2-y
            annotations.append({'id':len(annotations)+1,'image_id':r['image_id'],'category_id':1,
                                'bbox':[x,y,w,h],'area':w*h,'iscrowd':0})
        for p in r['predictions']:
            x,y,x2,y2=p['box']
            predictions.append({'image_id':r['image_id'],'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':p['score']})
    if not predictions:return {'mAP50':0.,'mAP50_95':0.}
    coco.dataset={'images':images,'annotations':annotations,'categories':[{'id':1,'name':'residuo'}],'info':{}}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex();detected=coco.loadRes(predictions)
        evaluation=COCOeval(coco,detected,'bbox')
        evaluation.evaluate();evaluation.accumulate();evaluation.summarize()
    return {'mAP50':float(evaluation.stats[1]),'mAP50_95':float(evaluation.stats[0])}



