"""Detector de residuos compacto multiobjeto entrenado desde cero (PyTorch).

Mejoras avanzadas de Visión por Computador:
1. Arquitectura con bloques residuales ligeros (ResBlocks) y fusión multiescala (FPN ligero)
   para combinar detalles de textura de bajo nivel (etapa 3, 32x32) con semántica profunda (etapa 4 y 5, 16x16).
2. Pérdida GIoU (Generalized Intersection over Union) combinada con Smooth L1 para optimizar
   directamente el solapamiento geométrico de cajas (IoU >= 0.5).
3. Focal Loss para presencia/objetividad con modulación gamma=2.0 y balance alpha para
   resolver el desbalance extremo de celdas vacías (fondo) frente a positivas.
4. Pérdida de clasificación con ponderación por frecuencia de clase y label smoothing.
5. Supresión de No-Máximos consciente de clase (Class-Aware / Batched NMS) para evitar
   que residuos de diferentes clases contiguos se supriman entre sí.
6. Evaluador con diagnóstico detallado de errores: Falsos Positivos de fondo, errores de
   localización (IoU < 0.5), confusiones de clase y omisiones (FN).
"""
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.annotation_storage import CLASSES, CLASS_DISPLAY, CLASS_TO_ID, ID_TO_CLASS

GRID_SIZE = 16  # Para imagen 256x256 -> paso 16 px por celda
NUM_CLASSES = len(CLASSES)  # 6 clases


class ConvBlock(nn.Module):
    """Bloque convolucional base con BatchNorm y LeakyReLU."""
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Bloque residual ligero con proyección residual si cambian los canales."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, kernel=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv2(self.conv1(x))
        return self.act(out + res)


class WasteDetectorNet(nn.Module):
    """Detector convolucional multiobjeto propio con bloques residuales y fusión multiescala."""
    def __init__(self, num_classes: int = NUM_CLASSES, grid_size: int = GRID_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size

        # Etapas de extracción convolucional
        self.stage1 = nn.Sequential(
            ConvBlock(3, 16, kernel=3, stride=2),  # 256 -> 128
            ConvBlock(16, 16, kernel=3)
        )
        self.stage2 = nn.Sequential(
            ConvBlock(16, 32, kernel=3, stride=2),  # 128 -> 64
            ResidualBlock(32, 32)
        )
        self.stage3 = nn.Sequential(
            ConvBlock(32, 64, kernel=3, stride=2),  # 64 -> 32
            ResidualBlock(64, 64)
        )
        self.stage4 = nn.Sequential(
            ConvBlock(64, 96, kernel=3, stride=2),  # 32 -> 16
            ResidualBlock(96, 96)
        )

        # Fusión multiescala: proyectar etapa 3 (32x32) a (16x16) y fusionar con etapa 4
        self.fuse_stage3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        # Etapa 5 con contexto fusionado: (96 + 32) = 128 canales
        self.stage5 = nn.Sequential(
            ResidualBlock(128, 128),
            ConvBlock(128, 96, kernel=1)
        )

        # Cabeza Desacoplada con Convoluciones Profundas para Discriminación de Materiales
        # (Recomendación del profesor: más convoluciones para desacoplar y diferenciar plástico vs vidrio)
        
        # Rama 1: Localización (presencia [1] + caja [4] = 5 canales)
        self.loc_head = nn.Sequential(
            ConvBlock(96, 64, kernel=3),
            ConvBlock(64, 64, kernel=3),
            nn.Conv2d(64, 5, kernel_size=1)
        )

        # Rama 2: Clasificación de Materiales Profunda (6 clases)
        # 3 capas convolucionales 3x3 adicionales para capturar micro-texturas, refracción y rigidez
        self.cls_head = nn.Sequential(
            ConvBlock(96, 96, kernel=3),
            ConvBlock(96, 96, kernel=3),
            ConvBlock(96, 64, kernel=3),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Inicialización Kaiming Normal (He) desde cero."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Sesgo inicial negativo de presencia para contrarrestar el desbalance de celdas de fondo
        final_loc_conv = self.loc_head[-1]
        if final_loc_conv.bias is not None:
            nn.init.constant_(final_loc_conv.bias[0], math.log(0.01 / 0.99))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)  # 32x32, 64 ch
        f4 = self.stage4(f3)  # 16x16, 96 ch

        # Fusión multiescala
        f3_proj = self.fuse_stage3(f3)  # 16x16, 32 ch
        f_cat = torch.cat([f4, f3_proj], dim=1)  # 16x16, 128 ch

        f5 = self.stage5(f_cat)  # 16x16, 96 ch
        loc = self.loc_head(f5)  # 16x16, 5 ch (obj + box)
        cls = self.cls_head(f5)  # 16x16, 6 ch (classes)

        # Concatenación en dimensión de canales: [B, 5 + 6, 16, 16] = [B, 11, 16, 16]
        return torch.cat([loc, cls], dim=1)


def calculate_giou_tensor(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calcula GIoU entre dos conjuntos de tensores de cajas (x1, y1, x2, y2).
    boxes1, boxes2: (N, 4)
    Retorna: (N,) con valores GIoU en [-1, 1].
    """
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)

    # Caja mínima envolvente (enclosing box C)
    c_x1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    c_y1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    c_x2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    c_y2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    c_area = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    giou = iou - (c_area - union).clamp(min=0) / c_area.clamp(min=1e-6)
    return giou


def torchvision_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thresh: float,
    class_indices: Optional[torch.Tensor] = None
) -> List[int]:
    """Non-Maximum Suppression (NMS) con soporte para separación por clases (Class-Aware NMS)."""
    if boxes.numel() == 0:
        return []

    # Si se pasan clases, aplicar desplazamiento por clase para que clases distintas no se supriman entre sí
    if class_indices is not None:
        max_coord = 4096.0
        offsets = class_indices.float().unsqueeze(1) * max_coord
        boxes_eval = boxes + offsets
    else:
        boxes_eval = boxes

    x1 = boxes_eval[:, 0]
    y1 = boxes_eval[:, 1]
    x2 = boxes_eval[:, 2]
    y2 = boxes_eval[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union.clamp(min=1e-6)

        below_thresh = (iou <= iou_thresh).nonzero(as_tuple=False).squeeze(1)
        order = order[below_thresh + 1]

    return keep


def decode_detections(
    raw_output: torch.Tensor,
    conf_threshold: float = 0.25,
    nms_iou_threshold: float = 0.45,
    img_size: int = 256,
    class_aware_nms: bool = True
) -> List[List[Dict[str, Any]]]:
    """Decodifica predicciones de la red con NMS consciente de clase."""
    B, C, H, W = raw_output.shape
    stride = img_size / H
    batch_results = []

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=raw_output.device),
        torch.arange(W, device=raw_output.device),
        indexing="ij"
    )
    grid_x = x_coords.float().unsqueeze(0)
    grid_y = y_coords.float().unsqueeze(0)

    for b in range(B):
        sample = raw_output[b]
        obj_prob = torch.sigmoid(sample[0])
        offset_x = torch.sigmoid(sample[1])
        offset_y = torch.sigmoid(sample[2])
        size_w = F.softplus(sample[3]) * 3.0
        size_h = F.softplus(sample[4]) * 3.0

        class_logits = sample[5:5 + NUM_CLASSES]
        class_probs = F.softmax(class_logits, dim=0)
        max_prob, class_idx = torch.max(class_probs, dim=0)

        # Confianza global calibrada
        scores = obj_prob * max_prob
        mask = scores >= conf_threshold

        if not mask.any():
            batch_results.append([])
            continue

        cx = (grid_x[0][mask] + offset_x[mask]) * stride
        cy = (grid_y[0][mask] + offset_y[mask]) * stride
        w = size_w[mask] * stride
        h = size_h[mask] * stride

        x1 = torch.clamp(cx - w / 2, 0, img_size)
        y1 = torch.clamp(cy - h / 2, 0, img_size)
        x2 = torch.clamp(cx + w / 2, 0, img_size)
        y2 = torch.clamp(cy + h / 2, 0, img_size)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        sub_scores = scores[mask]
        sub_classes = class_idx[mask]

        # NMS consciente de clase (permite múltiples residuos contiguos de distinta clase)
        keep = torchvision_nms(
            boxes, sub_scores, nms_iou_threshold,
            class_indices=sub_classes if class_aware_nms else None
        )

        # Supresión espacial inter-clase: un mismo objeto físico no puede tener dos clases simultáneas
        if len(keep) > 1:
            filtered_boxes = boxes[keep]
            filtered_scores = sub_scores[keep]
            spatial_keep = torchvision_nms(
                filtered_boxes, filtered_scores, iou_thresh=0.50,
                class_indices=None
            )
            keep = [keep[idx] for idx in spatial_keep]

        detections = []
        for k in keep:
            bx = boxes[k].cpu().tolist()
            sc = float(sub_scores[k].item())
            cid = int(sub_classes[k].item())
            cname = ID_TO_CLASS.get(cid, "desconocido")
            group = "Orgánico" if cname == "organicos" else "Inorgánico"
            detections.append({
                "box": (round(bx[0], 1), round(bx[1], 1), round(bx[2], 1), round(bx[3], 1)),
                "confidence": round(sc, 4),
                "class_id": cid,
                "class_name": cname,
                "group": group
            })
        batch_results.append(detections)

    return batch_results


def calculate_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Calcula el IoU escalar entre dos cajas (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 1e-6:
        return 0.0
    return inter / union


class DetectorEvaluator:
    """Evaluador riguroso con emparejamiento uno a uno bajo IoU >= umbral y diagnóstico de causas de error."""
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truths: List[List[Dict[str, Any]]],
        predictions: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        per_class_tp = {cls: 0 for cls in CLASSES}
        per_class_fp = {cls: 0 for cls in CLASSES}
        per_class_fn = {cls: 0 for cls in CLASSES}
        per_class_gt_count = {cls: 0 for cls in CLASSES}
        all_matched_ious = []

        # Diagnóstico de tipología de errores
        error_diagnostics = {
            "fp_background": 0,         # Predicción en el vacío (IoU < 0.1 con cualquier GT)
            "localization_error": 0,    # Clase correcta, pero 0.1 <= IoU < 0.5 (caja desviada)
            "class_confusion": 0,       # IoU >= 0.5 con un objeto real, pero clase equivocada
            "omissions_fn": 0           # Objeto real completamente ignorado
        }

        for img_idx, (gts, preds) in enumerate(zip(ground_truths, predictions)):
            for gt in gts:
                per_class_gt_count[gt["label"]] += 1

            matched_gt_indices = set()
            sorted_preds = sorted(enumerate(preds), key=lambda p: p[1]["confidence"], reverse=True)

            for pred_idx, pred in sorted_preds:
                best_iou = 0.0
                best_gt_idx = -1
                best_gt_class = None

                # Buscar la mejor coincidencia con cualquier GT
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gt_indices:
                        continue
                    gt_box = gt["box"] if "box" in gt else (gt["x1"], gt["y1"], gt["x2"], gt["y2"])
                    iou = calculate_iou(pred["box"], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_gt_class = gt["label"]

                if best_gt_idx != -1 and best_gt_class == pred["class_name"] and best_iou >= self.iou_threshold:
                    matched_gt_indices.add(best_gt_idx)
                    per_class_tp[pred["class_name"]] += 1
                    all_matched_ious.append(best_iou)
                else:
                    per_class_fp[pred["class_name"]] += 1
                    # Diagnosticar por qué falló
                    if best_iou < 0.1:
                        error_diagnostics["fp_background"] += 1
                    elif best_gt_class == pred["class_name"] and best_iou < self.iou_threshold:
                        error_diagnostics["localization_error"] += 1
                    elif best_iou >= self.iou_threshold and best_gt_class != pred["class_name"]:
                        error_diagnostics["class_confusion"] += 1
                    else:
                        error_diagnostics["fp_background"] += 1

            for gt_idx, gt in enumerate(gts):
                if gt_idx not in matched_gt_indices:
                    per_class_fn[gt["label"]] += 1
                    error_diagnostics["omissions_fn"] += 1

        results_per_class = {}
        for cls in CLASSES:
            tp = per_class_tp[cls]
            fp = per_class_fp[cls]
            fn = per_class_fn[cls]
            gt_n = per_class_gt_count[cls]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results_per_class[cls] = {
                "display_name": CLASS_DISPLAY[cls],
                "ground_truth_count": gt_n,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "evaluable": gt_n > 0
            }

        evaluable_classes = [c for c in CLASSES if results_per_class[c]["evaluable"]]
        mean_precision = (
            sum(results_per_class[c]["precision"] for c in evaluable_classes) / len(evaluable_classes)
            if evaluable_classes else 0.0
        )
        mean_recall = (
            sum(results_per_class[c]["recall"] for c in evaluable_classes) / len(evaluable_classes)
            if evaluable_classes else 0.0
        )
        mean_f1 = (
            sum(results_per_class[c]["f1_score"] for c in evaluable_classes) / len(evaluable_classes)
            if evaluable_classes else 0.0
        )

        unrepresented = [c for c in CLASSES if not results_per_class[c]["evaluable"]]

        return {
            "iou_threshold": self.iou_threshold,
            "classes": results_per_class,
            "mean_precision": round(mean_precision, 4),
            "mean_recall": round(mean_recall, 4),
            "mean_f1": round(mean_f1, 4),
            "avg_matched_iou": round(float(np.mean(all_matched_ious)), 4) if all_matched_ious else 0.0,
            "unrepresented_classes": [CLASS_DISPLAY[c] for c in unrepresented],
            "error_diagnostics": error_diagnostics
        }
