"""Entrenamiento y evaluación del detector de residuos desde cero.

Optimizaciones de Visión por Computador Senior:
1. Aumento de datos seguro multiobjeto (volteo horizontal, jitter fotométrico y escalado afín seguro).
2. Pérdida Focal Loss para presencia (mitiga el desbalance de celdas de fondo).
3. Pérdida de regresión híbrida GIoU + Smooth L1 evaluada en píxeles absolutos.
4. Pérdida de clasificación con ponderación inversa de clases y label smoothing.
5. Diagnóstico de errores categorizados (falsos positivos de fondo, localización < 0.5, confusión de clase y omisiones).
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.annotation_storage import AnnotationDataset, CLASSES, CLASS_TO_ID
from training.multi_waste_detector import (
    WasteDetectorNet, decode_detections, DetectorEvaluator, calculate_giou_tensor,
    GRID_SIZE, NUM_CLASSES
)
from utils.visual_comparison import save_comparison


class SafeDataAugmentor:
    """Aumento de datos geométrico y fotométrico con preservación estricta de coordenadas."""
    def __init__(self, img_size: int = 256):
        self.img_size = img_size

    def __call__(self, img: np.ndarray, boxes: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        h, w = img.shape[:2]
        out_img = img.copy()
        out_boxes = [dict(b) for b in boxes]

        # 1. Volteo horizontal aleatorio (50% prob)
        if random.random() < 0.5:
            out_img = np.fliplr(out_img).copy()
            for b in out_boxes:
                new_x1 = w - b["x2"]
                new_x2 = w - b["x1"]
                b["x1"] = max(0.0, float(new_x1))
                b["x2"] = min(float(w), float(new_x2))

        # 2. Jitter fotométrico (brillo, contraste y saturación moderados)
        if random.random() < 0.7:
            # Factor de brillo [0.85, 1.15]
            alpha_b = random.uniform(0.85, 1.15)
            # Factor de contraste [0.85, 1.15]
            alpha_c = random.uniform(0.85, 1.15)

            hsv = cv2.cvtColor(out_img, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Saturación
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            # Brillo (V)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * alpha_b, 0, 255)
            out_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Contraste
            mean_val = np.mean(out_img, axis=(0, 1), keepdims=True)
            out_img = np.clip((out_img - mean_val) * alpha_c + mean_val, 0, 255).astype(np.uint8)

        # 2.1 Aumento especializado para Vidrio y Plástico (Reflejos Especulares y Rigidez vs Deformación)
        # El vidrio presenta reflejos especulares de alta intensidad y rigidez geométrica.
        # El plástico presenta reflexiones difusas y arrugas/deformaciones elásticas.
        for b in out_boxes:
            lbl = b.get("label", "")
            bx1, by1, bx2, by2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            bw, bh = bx2 - bx1, by2 - by1
            if bw > 16 and bh > 16:
                if lbl == "vidrio" and random.random() < 0.5:
                    # Simular pico especular reflectivo blanco puro en vidrio (n~1.52)
                    sx = random.randint(bx1 + int(bw * 0.2), bx1 + int(bw * 0.8))
                    sy = random.randint(by1 + int(bh * 0.2), by1 + int(bh * 0.8))
                    sr = max(2, min(bw, bh) // 8)
                    cv2.circle(out_img, (sx, sy), sr, (255, 255, 255), -1)
                elif lbl == "plastico" and random.random() < 0.4:
                    # Simular pliegues o sombras difusas típicas de botellas PET recicladas
                    px1 = random.randint(bx1 + 2, bx1 + max(3, bw - 10))
                    py1 = random.randint(by1 + 2, by1 + max(3, bh - 10))
                    px2 = min(bx2 - 2, px1 + random.randint(6, 18))
                    py2 = min(by2 - 2, py1 + random.randint(2, 6))
                    out_img[py1:py2, px1:px2] = np.clip(out_img[py1:py2, px1:px2].astype(np.float32) * 0.75, 0, 255).astype(np.uint8)

        # 3. Escalado y traslación segura (40% prob)
        if random.random() < 0.4 and len(out_boxes) > 0:
            scale = random.uniform(0.92, 1.08)
            dx = random.uniform(-10, 10)
            dy = random.uniform(-10, 10)

            M = np.float32([[scale, 0, (1 - scale) * (w / 2) + dx],
                            [0, scale, (1 - scale) * (h / 2) + dy]])
            transformed_img = cv2.warpAffine(out_img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            candidate_boxes = []
            all_valid = True
            for b in out_boxes:
                nx1 = b["x1"] * scale + M[0, 2]
                ny1 = b["y1"] * scale + M[1, 2]
                nx2 = b["x2"] * scale + M[0, 2]
                ny2 = b["y2"] * scale + M[1, 2]

                cx1 = max(0.0, min(float(w), nx1))
                cy1 = max(0.0, min(float(h), ny1))
                cx2 = max(0.0, min(float(w), nx2))
                cy2 = max(0.0, min(float(h), ny2))

                orig_area = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
                new_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)

                if (cx2 - cx1) >= 8 and (cy2 - cy1) >= 8 and new_area >= 0.4 * orig_area:
                    candidate_boxes.append({
                        "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                        "label": b["label"]
                    })
                else:
                    all_valid = False
                    break

            if all_valid and len(candidate_boxes) == len(out_boxes):
                out_img = transformed_img
                out_boxes = candidate_boxes

        return out_img, out_boxes


class WasteDataset(Dataset):
    """Dataset para entrenamiento y validación con aumento de datos en train."""
    def __init__(self, records: List[Any], img_size: int = 256, is_train: bool = True):
        self.records = records
        self.img_size = img_size
        self.is_train = is_train
        self.augmentor = SafeDataAugmentor(img_size) if is_train else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec.image_path

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            scale_x = self.img_size / max(orig_w, 1)
            scale_y = self.img_size / max(orig_h, 1)
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
        else:
            img_resized = np.full((self.img_size, self.img_size, 3), 120, dtype=np.uint8)
            scale_x = scale_y = 1.0

        current_boxes = []
        for b in rec.boxes:
            current_boxes.append({
                "x1": b["x1"] * scale_x,
                "y1": b["y1"] * scale_y,
                "x2": b["x2"] * scale_x,
                "y2": b["y2"] * scale_y,
                "label": b["label"]
            })

        # Aplicar aumento seguro solo en entrenamiento
        if self.is_train and self.augmentor is not None:
            img_resized, current_boxes = self.augmentor(img_resized, current_boxes)

        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0

        stride = self.img_size / GRID_SIZE
        target_obj = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
        # target_box: [x_offset, y_offset, w_norm, h_norm]
        target_box = torch.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=torch.float32)
        # target_boxes_px: [x1, y1, x2, y2] en píxeles absolutos para pérdida GIoU
        target_boxes_px = torch.zeros((GRID_SIZE, GRID_SIZE, 4), dtype=torch.float32)
        target_cls = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.long)

        raw_boxes_list = []
        for b in current_boxes:
            x1 = float(np.clip(b["x1"], 0, self.img_size))
            y1 = float(np.clip(b["y1"], 0, self.img_size))
            x2 = float(np.clip(b["x2"], 0, self.img_size))
            y2 = float(np.clip(b["y2"], 0, self.img_size))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(x2 - x1, 4.0)
            bh = max(y2 - y1, 4.0)
            cls_name = b["label"]
            cls_id = CLASS_TO_ID.get(cls_name, 0)

            raw_boxes_list.append({"box": (x1, y1, x2, y2), "label": cls_name})

            gx = int(np.clip(cx / stride, 0, GRID_SIZE - 1))
            gy = int(np.clip(cy / stride, 0, GRID_SIZE - 1))

            target_obj[gy, gx] = 1.0
            target_box[gy, gx, 0] = (cx / stride) - gx
            target_box[gy, gx, 1] = (cy / stride) - gy
            target_box[gy, gx, 2] = bw / (stride * 3.0)
            target_box[gy, gx, 3] = bh / (stride * 3.0)
            target_boxes_px[gy, gx] = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
            target_cls[gy, gx] = cls_id

        return tensor, target_obj, target_box, target_boxes_px, target_cls, raw_boxes_list, img_resized


def detector_loss(
    pred_raw: torch.Tensor,
    target_obj: torch.Tensor,
    target_box: torch.Tensor,
    target_boxes_px: torch.Tensor,
    target_cls: torch.Tensor,
    img_size: int = 256,
    class_weights: Optional[torch.Tensor] = None
):
    """Pérdida multiobjetivo con Focal Loss, GIoU directo y Cross-Entropy ponderada."""
    B, _, H, W = pred_raw.shape
    stride = img_size / H
    pred_obj = pred_raw[:, 0]
    pred_box = pred_raw[:, 1:5]
    pred_cls = pred_raw[:, 5:5 + NUM_CLASSES]

    # 1. Focal Loss para presencia orientada a alto Recall
    # Se incrementa la ponderación positiva (alpha=0.85, gamma=1.5) para penalizar omisiones (FN)
    prob_obj = torch.sigmoid(pred_obj).clamp(1e-5, 1.0 - 1e-5)
    focal_pos = -0.85 * ((1.0 - prob_obj) ** 1.5) * torch.log(prob_obj) * target_obj
    focal_neg = -0.15 * (prob_obj ** 2.0) * torch.log(1.0 - prob_obj) * (1.0 - target_obj)
    pos_cells_total = max(1.0, float(target_obj.sum().item()))
    loss_obj = (focal_pos + focal_neg).sum() / pos_cells_total

    pos_mask = (target_obj > 0.5)
    pos_count = pos_mask.sum().item()

    if pos_count > 0:
        pos_box_pred = pred_box.permute(0, 2, 3, 1)[pos_mask]
        pos_box_target = target_box[pos_mask]
        pos_boxes_target_px = target_boxes_px[pos_mask]

        # Coordenadas relativas
        p_x = torch.sigmoid(pos_box_pred[:, 0])
        p_y = torch.sigmoid(pos_box_pred[:, 1])
        p_w = F.softplus(pos_box_pred[:, 2])
        p_h = F.softplus(pos_box_pred[:, 3])

        # Smooth L1 en parámetros geométricos
        pred_geom = torch.stack([p_x, p_y, p_w, p_h], dim=1)
        loss_l1 = F.smooth_l1_loss(pred_geom, pos_box_target, reduction="mean")

        # 2. Pérdida GIoU evaluada en píxeles absolutos
        # Obtener coordenadas de cuadrícula (grid_x, grid_y) para las celdas positivas
        grid_indices = pos_mask.nonzero(as_tuple=False)  # (K, 3): [batch_idx, y_idx, x_idx]
        gy = grid_indices[:, 1].float()
        gx = grid_indices[:, 2].float()

        pred_cx_px = (gx + p_x) * stride
        pred_cy_px = (gy + p_y) * stride
        pred_w_px = p_w * stride * 3.0
        pred_h_px = p_h * stride * 3.0

        pred_boxes_px = torch.stack([
            pred_cx_px - pred_w_px / 2.0,
            pred_cy_px - pred_h_px / 2.0,
            pred_cx_px + pred_w_px / 2.0,
            pred_cy_px + pred_h_px / 2.0
        ], dim=1)

        giou_vals = calculate_giou_tensor(pred_boxes_px, pos_boxes_target_px)
        loss_giou = (1.0 - giou_vals).mean()

        # Refuerzo de localización GIoU (peso 2.5) para maximizar cajas con IoU >= 0.50
        loss_box = 2.5 * loss_giou + 0.5 * loss_l1

        # 3. Clasificación con Cross-Entropy ponderada y label smoothing suave
        pos_cls_pred = pred_cls.permute(0, 2, 3, 1)[pos_mask]
        pos_cls_target = target_cls[pos_mask]
        loss_cls = F.cross_entropy(
            pos_cls_pred, pos_cls_target,
            weight=class_weights,
            label_smoothing=0.02,
            reduction="mean"
        )
    else:
        loss_box = torch.tensor(0.0, device=pred_raw.device)
        loss_cls = torch.tensor(0.0, device=pred_raw.device)

    total_loss = loss_obj + 2.5 * loss_box + 1.0 * loss_cls
    return total_loss, {
        "loss_obj": float(loss_obj.item()),
        "loss_box": float(loss_box.item()) if pos_count > 0 else 0.0,
        "loss_cls": float(loss_cls.item()) if pos_count > 0 else 0.0,
        "positive_cells": pos_count
    }


def collate_fn(batch):
    tensors = torch.stack([b[0] for b in batch])
    objs = torch.stack([b[1] for b in batch])
    boxes = torch.stack([b[2] for b in batch])
    boxes_px = torch.stack([b[3] for b in batch])
    clss = torch.stack([b[4] for b in batch])
    raw_boxes = [b[5] for b in batch]
    raw_imgs = [b[6] for b in batch]
    return tensors, objs, boxes, boxes_px, clss, raw_boxes, raw_imgs


def train_detector(
    db_path: str = "data/annotations/annotations.json",
    output_dir: str = "runs/waste_detector",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001,
    seed: int = 42,
    technical_check: bool = False
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Entrenador] Dispositivo seleccionado: {device}")
    if device.type == "cuda":
        print(f"[Entrenador] GPU: {torch.cuda.get_device_name(0)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = AnnotationDataset(db_path)
    records = list(dataset.records.values())

    if technical_check or len(records) == 0:
        print("[Entrenador] MODO PRUEBA TÉCNICA: Creando datos sintéticos de prueba para verificar pipeline completo...")
        from utils.annotation_storage import AnnotationRecord
        records = [
            AnnotationRecord(
                image_path=f"synthetic_{i}.jpg",
                video_id="video_tech_test",
                session_id="session_tech_01",
                frame_index=i,
                boxes=[
                    {"x1": 30.0, "y1": 40.0, "x2": 110.0, "y2": 160.0, "label": "plastico"},
                    {"x1": 140.0, "y1": 80.0, "x2": 210.0, "y2": 180.0, "label": "metal"}
                ] if i % 4 != 0 else [],
                is_empty_reviewed=(i % 4 == 0),
                split="train" if i < 16 else ("val" if i < 20 else "test"),
                width=256,
                height=256
            )
            for i in range(24)
        ]

    train_recs = [r for r in records if r.split == "train"]
    val_recs = [r for r in records if r.split == "val"]
    test_recs = [r for r in records if r.split == "test"]

    if not val_recs:
        val_recs = train_recs[:max(1, len(train_recs) // 5)]
    if not test_recs:
        test_recs = val_recs

    # Calcular pesos de clase con refuerzo inverso para clases minoritarias / bajo recall
    class_counts = [0] * NUM_CLASSES
    for r in train_recs:
        for b in r.boxes:
            cid = CLASS_TO_ID.get(b["label"], 0)
            class_counts[cid] += 1
    total_objects = max(1, sum(class_counts))
    # Reforzar vidrio, metal y plástico con exponente 0.6 para suavizar el dominio extremo de orgánicos
    weights = [(total_objects / max(1, c)) ** 0.6 for c in class_counts]
    norm_factor = sum(weights) / NUM_CLASSES
    weights = [w / norm_factor for w in weights]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    train_ds = WasteDataset(train_recs, is_train=True)
    val_ds = WasteDataset(val_recs, is_train=False)
    test_ds = WasteDataset(test_recs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = WasteDetectorNet(num_classes=NUM_CLASSES, grid_size=GRID_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"[Entrenador] Arquitectura: WasteDetectorNet (ResBlocks + FPN Ligero) | Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Entrenador] Train: {len(train_recs)} | Val: {len(val_recs)} | Test: {len(test_recs)}")

    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float("inf")
    best_checkpoint_path = out_path / "best_model.pth"

    start_total = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_epoch = 0.0

        for tensors, objs, boxes, boxes_px, clss, _, _ in train_loader:
            tensors = tensors.to(device)
            objs = objs.to(device)
            boxes = boxes.to(device)
            boxes_px = boxes_px.to(device)
            clss = clss.to(device)

            optimizer.zero_grad()
            preds = model(tensors)
            loss, _ = detector_loss(preds, objs, boxes, boxes_px, clss, img_size=256, class_weights=class_weights_tensor)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= max(1, len(train_loader))
        scheduler.step()

        # Validación
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for tensors, objs, boxes, boxes_px, clss, _, _ in val_loader:
                tensors = tensors.to(device)
                objs = objs.to(device)
                boxes = boxes.to(device)
                boxes_px = boxes_px.to(device)
                clss = clss.to(device)
                preds = model(tensors)
                loss, _ = detector_loss(preds, objs, boxes, boxes_px, clss, img_size=256, class_weights=class_weights_tensor)
                val_loss_epoch += loss.item()
        val_loss_epoch /= max(1, len(val_loader))

        ep_duration = time.time() - t0
        history["train_loss"].append(round(train_loss_epoch, 4))
        history["val_loss"].append(round(val_loss_epoch, 4))
        history["epoch_times"].append(round(ep_duration, 2))

        print(f"Época [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} ({ep_duration:.1f}s)")

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss_epoch,
                "architecture": "WasteDetectorNet-ResFPN",
                "grid_size": GRID_SIZE,
                "num_classes": NUM_CLASSES,
                "classes": CLASSES,
                "pretrained": False,
                "seed": seed
            }, best_checkpoint_path)

    total_training_time = time.time() - start_total
    print(f"\n[Entrenador] Entrenamiento completado en {total_training_time:.1f}s. Checkpoint óptimo guardado en: {best_checkpoint_path}")

    # Evaluación y diagnóstico de errores en Test
    print("\n" + "=" * 68)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA RESERVADO (IoU >= 0.5)")
    print("=" * 68)

    best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()

    evaluator = DetectorEvaluator(iou_threshold=0.5)
    test_gts = []
    test_preds = []

    comp_dir = out_path / "eval_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(test_ds)):
            tensor, _, _, _, _, raw_boxes, raw_img = test_ds[idx]
            batch_tensor = tensor.unsqueeze(0).to(device)
            raw_out = model(batch_tensor)
            preds = decode_detections(
                raw_out, conf_threshold=0.20, nms_iou_threshold=0.45, img_size=256, class_aware_nms=True
            )[0]

            test_gts.append(raw_boxes)
            test_preds.append(preds)

            if idx < 5:
                save_comparison(
                    image_rgb=raw_img,
                    predictions=preds,
                    ground_truths=raw_boxes,
                    output_path=str(comp_dir / f"test_comparison_{idx:03d}.jpg")
                )

    metrics = evaluator.evaluate(test_gts, test_preds)

    print(f"Precisión Media (Mean Precision): {metrics['mean_precision'] * 100:.2f}%")
    print(f"Exhaustividad Media (Mean Recall): {metrics['mean_recall'] * 100:.2f}%")
    print(f"F1-Score Medio (Macro-F1)       : {metrics['mean_f1'] * 100:.2f}%")
    print(f"IoU promedio de emparejados     : {metrics['avg_matched_iou']:.3f}")
    print("-" * 68)
    print(f"{'Clase':<16} {'GTs':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 68)
    for cls in CLASSES:
        cm = metrics["classes"][cls]
        print(f"{cm['display_name']:<16} {cm['ground_truth_count']:>5} {cm['TP']:>5} {cm['FP']:>5} {cm['FN']:>5} "
              f"{cm['precision'] * 100:>7.1f}% {cm['recall'] * 100:>7.1f}% {cm['f1_score'] * 100:>7.1f}%")

    # Diagnóstico de errores
    errs = metrics.get("error_diagnostics", {})
    print("-" * 68)
    print("DIAGNÓSTICO DETALLADO DE ERRORES:")
    print(f"  • Falsos positivos de fondo (IoU < 0.1) : {errs.get('fp_background', 0)}")
    print(f"  • Cajas mal localizadas (0.1 <= IoU < 0.5): {errs.get('localization_error', 0)}")
    print(f"  • Confusiones de clase (IoU >= 0.5)      : {errs.get('class_confusion', 0)}")
    print(f"  • Omisiones / No detectados (FN)        : {errs.get('omissions_fn', 0)}")
    print("=" * 68)

    with open(out_path / "test_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Entrenar detector de residuos optimizado desde cero")
    parser.add_argument("--db", type=str, default="data/annotations/annotations.json")
    parser.add_argument("--out", type=str, default="runs/waste_detector")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--technical_check", action="store_true", help="Ejecutar prueba técnica con datos sintéticos")
    args = parser.parse_args()

    train_detector(
        db_path=args.db,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        technical_check=args.technical_check
    )


if __name__ == "__main__":
    main()
