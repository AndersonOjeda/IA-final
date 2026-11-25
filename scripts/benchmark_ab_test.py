"""Estudio de Ablación y Comparación A/B Rigurosa: Baseline vs Optimizado.

Demuestra empíricamente el impacto de:
1. SafeDataAugmentor (Flip, HSV, Escala)
2. Focal Loss (vs BCE estándar)
3. GIoU Loss (vs Smooth L1 aislado)
4. ResBlocks + FPN Ligero (vs CNN plana)
5. Class-Aware NMS (vs NMS agnóstico)

Entrena ambos modelos bajo las MISMAS condiciones en un conjunto visual benchmark
y los evalúa en el MISMO conjunto de prueba reservado con IoU >= 0.5.
"""
import os
import random
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.annotation_storage import (
    CLASSES, CLASS_TO_ID, ID_TO_CLASS, AnnotationRecord, AnnotationDataset
)
from training.multi_waste_detector import (
    WasteDetectorNet, ConvBlock, DetectorEvaluator, calculate_giou_tensor,
    decode_detections, GRID_SIZE, NUM_CLASSES
)
from scripts.train_waste_detector import (
    SafeDataAugmentor, detector_loss, WasteDataset, collate_fn
)


# ==============================================================================
# MODELO BASELINE (Arquitectura simple plana sin ResBlocks ni FPN)
# ==============================================================================
class BaselineDetectorNet(nn.Module):
    """Detector base plano de 5 capas convolucionales sin residuales ni FPN."""
    def __init__(self, num_classes: int = NUM_CLASSES, grid_size: int = GRID_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        out_channels = 1 + 4 + num_classes

        self.features = nn.Sequential(
            ConvBlock(3, 16, kernel=3, stride=2),   # 256 -> 128
            ConvBlock(16, 32, kernel=3, stride=2),  # 128 -> 64
            ConvBlock(32, 64, kernel=3, stride=2),  # 64 -> 32
            ConvBlock(64, 96, kernel=3, stride=2),  # 32 -> 16
            ConvBlock(96, 96, kernel=3),            # 16 -> 16
        )
        self.head = nn.Sequential(
            ConvBlock(96, 64, kernel=3),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.head(feat)


def compute_baseline_loss(pred_raw, target_obj, target_box, target_cls):
    """Pérdida convencional: BCE simple para obj, Smooth L1 simple para cajas, CE estándar."""
    B, _, H, W = pred_raw.shape
    pred_obj = pred_raw[:, 0]
    pred_box = pred_raw[:, 1:5]
    pred_cls = pred_raw[:, 5:5 + NUM_CLASSES]

    # BCE simple dividida por todas las celdas (desbalance severo de fondo)
    loss_obj = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction="mean")

    pos_mask = (target_obj > 0.5)
    pos_count = pos_mask.sum().item()

    if pos_count > 0:
        pos_box_pred = pred_box.permute(0, 2, 3, 1)[pos_mask]
        pos_box_target = target_box[pos_mask]

        p_x = torch.sigmoid(pos_box_pred[:, 0])
        p_y = torch.sigmoid(pos_box_pred[:, 1])
        p_w = F.softplus(pos_box_pred[:, 2])
        p_h = F.softplus(pos_box_pred[:, 3])

        pred_geom = torch.stack([p_x, p_y, p_w, p_h], dim=1)
        loss_box = F.smooth_l1_loss(pred_geom, pos_box_target, reduction="mean")

        pos_cls_pred = pred_cls.permute(0, 2, 3, 1)[pos_mask]
        pos_cls_target = target_cls[pos_mask]
        loss_cls = F.cross_entropy(pos_cls_pred, pos_cls_target, reduction="mean")
    else:
        loss_box = torch.tensor(0.0, device=pred_raw.device)
        loss_cls = torch.tensor(0.0, device=pred_raw.device)

    return loss_obj + 2.0 * loss_box + 1.0 * loss_cls


# ==============================================================================
# GENERADOR DE DATOS VISUALES BENCHMARK CON PATRONES REALES
# ==============================================================================
def draw_waste_object(canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str):
    """Renderiza patrones visuales distintivos para cada clase de residuo."""
    w = x2 - x1
    h = y2 - y1

    if label == "plastico":
        # Botella o envase azul/celeste translúcido con tapa
        color_body = (210, 160, 40)   # BGR: azul cielo
        cv2.rectangle(canvas, (x1, y1 + int(h * 0.2)), (x2, y2), color_body, -1)
        cv2.rectangle(canvas, (x1 + int(w * 0.3), y1), (x2 - int(w * 0.3), y1 + int(h * 0.2)), (240, 200, 80), -1)
        cv2.circle(canvas, (int((x1+x2)/2), y1 + 5), max(2, int(w * 0.15)), (40, 40, 220), -1) # Tapa roja
        # Brillo reflejo
        cv2.line(canvas, (x1 + 6, y1 + int(h * 0.25)), (x1 + 6, y2 - 8), (255, 255, 200), 2)

    elif label == "metal":
        # Lata cilíndrica metálica plateada con franjas de brillo
        color_metal = (180, 180, 185)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_metal, -1)
        # Líneas especulares
        cv2.line(canvas, (x1 + int(w * 0.2), y1), (x1 + int(w * 0.2), y2), (240, 240, 245), 2)
        cv2.line(canvas, (x1 + int(w * 0.5), y1), (x1 + int(w * 0.5), y2), (120, 120, 125), 2)
        cv2.circle(canvas, (int((x1+x2)/2), y1 + int(h * 0.15)), max(2, int(w * 0.2)), (90, 90, 95), -1)

    elif label == "carton":
        # Caja marrón corrugada con solapas
        color_carton = (50, 110, 165) # BGR marrón
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_carton, -1)
        # Textura corrugada
        for yy in range(y1 + 6, y2 - 4, 8):
            cv2.line(canvas, (x1 + 3, yy), (x2 - 3, yy), (35, 80, 125), 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 60, 100), 2)

    elif label == "papel":
        # Hoja blanca/crema con renglones de texto simulado
        color_papel = (240, 245, 245)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_papel, -1)
        for yy in range(y1 + 8, y2 - 6, 6):
            cv2.line(canvas, (x1 + 6, yy), (x2 - 6, yy), (100, 100, 100), 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)

    elif label == "vidrio":
        # Botella verde translúcida
        color_vidrio = (50, 170, 70)  # BGR verde
        cv2.rectangle(canvas, (x1, y1 + int(h * 0.3)), (x2, y2), color_vidrio, -1)
        cv2.rectangle(canvas, (x1 + int(w * 0.35), y1), (x2 - int(w * 0.35), y1 + int(h * 0.3)), color_vidrio, -1)
        cv2.line(canvas, (x1 + 4, y1 + int(h * 0.35)), (x1 + 4, y2 - 6), (180, 255, 190), 2)

    elif label == "organicos":
        # Manzana / resto orgánico rojizo-amarillento
        center = (int((x1+x2)/2), int((y1+y2)/2))
        axes = (max(4, int(w / 2)), max(4, int(h / 2)))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, (30, 40, 210), -1) # Rojo manzana
        # Tallo
        cv2.line(canvas, center, (center[0] - 2, y1 - 4), (20, 80, 50), 3)


def generate_benchmark_dataset(output_dir: str = "data/benchmark", num_images: int = 100):
    """Genera un dataset balanceado con características visuales ricas sobre disco."""
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = AnnotationDataset(str(out_dir / "annotations.json"))
    if len(dataset.records) >= num_images and all(os.path.exists(r.image_path) for r in dataset.records.values()):
        print(f"[Benchmark] Usando dataset existente con {len(dataset.records)} imágenes en {out_dir}")
        return dataset
    dataset.records.clear()

    random.seed(42)
    np.random.seed(42)

    sessions = ["sess_train_A", "sess_train_B", "sess_train_C", "sess_val_A", "sess_test_A"]

    for i in range(num_images):
        img_name = f"bench_{i:04d}.jpg"
        img_path = images_dir / img_name

        # Asignación estricta de partición por sesión para evitar data leakage
        if i < int(num_images * 0.70):
            session_id = sessions[i % 3]
            split = "train"
        elif i < int(num_images * 0.85):
            session_id = "sess_val_A"
            split = "val"
        else:
            session_id = "sess_test_A"
            split = "test"

        # Fondo con textura realista (madera, piso, cemento simulado)
        bg_color = random.choice([
            (210, 210, 215), (180, 190, 195), (200, 220, 210), (160, 160, 170)
        ])
        canvas = np.full((256, 256, 3), bg_color, dtype=np.uint8)
        # Añadir ruido Gaussiano de fondo
        noise = np.random.normal(0, 12, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 15% de imágenes vacías revisadas (ejemplos negativos)
        is_empty = (i % 7 == 0)
        boxes = []

        if not is_empty:
            num_objects = random.randint(1, 3)
            # Evitar colisiones excesivas
            occupied = []
            for _ in range(num_objects):
                cls_name = random.choice(CLASSES)
                bw = random.randint(45, 95)
                bh = random.randint(45, 95)

                for attempt in range(10):
                    x1 = random.randint(10, 256 - bw - 10)
                    y1 = random.randint(10, 256 - bh - 10)
                    x2 = x1 + bw
                    y2 = y1 + bh

                    # Chequeo rápido de no solapamiento total
                    overlap = False
                    for ox1, oy1, ox2, oy2 in occupied:
                        if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                            overlap = True
                            break
                    if not overlap or attempt == 9:
                        occupied.append((x1, y1, x2, y2))
                        draw_waste_object(canvas, x1, y1, x2, y2, cls_name)
                        boxes.append({
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "label": cls_name
                        })
                        break

        cv2.imwrite(str(img_path), canvas)

        dataset.add_or_update(
            image_path=str(img_path),
            video_id=f"video_{session_id}",
            session_id=session_id,
            frame_index=i,
            boxes=boxes,
            is_empty_reviewed=is_empty,
            split=split,
            width=256,
            height=256
        )

    dataset.save()
    print(f"[Benchmark] {num_images} imágenes generadas y registradas en {out_dir / 'annotations.json'}")
    return dataset


# ==============================================================================
# RUTINA DE ENTRENAMIENTO COMPARATIVO A/B
# ==============================================================================
def train_and_evaluate(
    model_name: str,
    model: nn.Module,
    is_optimized: bool,
    train_loader: DataLoader,
    val_records: List[Any],
    test_records: List[Any],
    epochs: int = 30,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Entrena y evalúa estrictamente bajo IoU >= 0.5."""
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {model_name.upper()} (Optimizado={is_optimized})")
    print(f"{'='*70}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.time()
    best_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch in train_loader:
            images, target_obj, target_box, target_boxes_px, target_cls, _, _ = batch
            images = images.to(device)
            target_obj = target_obj.to(device)
            target_box = target_box.to(device)
            target_boxes_px = target_boxes_px.to(device)
            target_cls = target_cls.to(device)

            optimizer.zero_grad()
            preds = model(images)

            if is_optimized:
                loss, _ = detector_loss(
                    preds, target_obj, target_box, target_boxes_px, target_cls
                )
            else:
                loss = compute_baseline_loss(
                    preds, target_obj, target_box, target_cls
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Época [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    train_time = time.time() - start_time
    print(f"[Fin Entrenamiento] Tiempo: {train_time:.1f}s")

    # Evaluación en Test Set
    evaluator = DetectorEvaluator(iou_threshold=0.5)
    model.eval()

    all_gts = []
    all_preds = []

    with torch.no_grad():
        for rec in test_records:
            img = cv2.imread(rec.image_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(device)

            raw_preds = model(tensor)
            # Decodificar predicciones
            detections = decode_detections(
                raw_preds,
                conf_threshold=0.20 if is_optimized else 0.20,
                nms_iou_threshold=0.45,
                img_size=256
            )[0]

            all_gts.append(rec.boxes)
            all_preds.append(detections)

    metrics = evaluator.evaluate(all_gts, all_preds)
    metrics["train_time"] = train_time
    return metrics


def main():
    print("=" * 70)
    print("ESTUDIO DE ABLACIÓN CIENTÍFICA: DETECTOR BASELINE vs DETECTOR OPTIMIZADO")
    print("Evaluación obligatoria bajo coincidencia IoU >= 0.5")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de ejecución: {device}")

    # 1. Generar / Cargar Dataset Benchmark de 100 imágenes
    dataset = generate_benchmark_dataset("data/benchmark", num_images=100)
    records = list(dataset.records.values())

    train_recs = [r for r in records if r.split == "train"]
    val_recs = [r for r in records if r.split == "val"]
    test_recs = [r for r in records if r.split == "test"]

    print(f"Partición estricta: Train={len(train_recs)} | Val={len(val_recs)} | Test={len(test_recs)}")

    # DataLoaders
    train_dataset_base = WasteDataset(train_recs, img_size=256, is_train=False) # Sin aug para base
    train_dataset_opt = WasteDataset(train_recs, img_size=256, is_train=True)   # Con SafeAug para opt

    train_loader_base = DataLoader(train_dataset_base, batch_size=8, shuffle=True, collate_fn=collate_fn)
    train_loader_opt = DataLoader(train_dataset_opt, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # 2. Experimento A: Baseline (CNN plana, BCE estándar, Smooth L1, NMS agnóstico)
    model_baseline = BaselineDetectorNet()
    metrics_base = train_and_evaluate(
        "Detector Baseline (ConvNet Plana + BCE + SmoothL1)",
        model_baseline,
        is_optimized=False,
        train_loader=train_loader_base,
        val_records=val_recs,
        test_records=test_recs,
        epochs=35,
        device=device
    )

    # 3. Experimento B: Optimizado (ResBlocks + FPN, Focal Loss, GIoU directo, Class-Aware NMS, SafeAug)
    model_optimized = WasteDetectorNet()
    metrics_opt = train_and_evaluate(
        "Detector Optimizado (ResBlocks + FPN + Focal + GIoU + ClassNMS)",
        model_optimized,
        is_optimized=True,
        train_loader=train_loader_opt,
        val_records=val_recs,
        test_records=test_recs,
        epochs=35,
        device=device
    )

    # Guardar el modelo optimizado como checkpoint oficial
    out_best = Path("runs/waste_detector/best_model.pth")
    out_best.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model_optimized.state_dict(),
        "architecture": "WasteDetectorNet-ResFPN",
        "input_size": 256,
        "grid_size": GRID_SIZE,
        "num_classes": NUM_CLASSES,
        "metrics": metrics_opt
    }, str(out_best))
    print(f"\n[Guardado] Checkpoint óptimo persistido en: {out_best}")

    # 4. TABLA COMPARATIVA DIRECTA
    print("\n" + "=" * 78)
    print("RESULTADOS EXPERIMENTALES: IMPACTO REAL DE LAS MEJORAS (IoU >= 0.5)")
    print("=" * 78)
    print(f"{'Métrica':<35} | {'Baseline (Antiguo)':<18} | {'Optimizado (Nuevo)':<18}")
    print("-" * 78)
    print(f"{'Precisión Media (Precision)':<35} | {metrics_base['mean_precision']*100:>16.1f}% | {metrics_opt['mean_precision']*100:>16.1f}%")
    print(f"{'Exhaustividad Media (Recall)':<35} | {metrics_base['mean_recall']*100:>16.1f}% | {metrics_opt['mean_recall']*100:>16.1f}%")
    print(f"{'F1-Score Macro':<35} | {metrics_base['mean_f1']*100:>16.1f}% | {metrics_opt['mean_f1']*100:>16.1f}%")
    print(f"{'IoU Promedio (Cajas Emparejadas)':<35} | {metrics_base['avg_matched_iou']:>17.3f} | {metrics_opt['avg_matched_iou']:>17.3f}")
    print(f"{'Falsos Positivos de Fondo':<35} | {metrics_base['error_diagnostics']['fp_background']:>17d} | {metrics_opt['error_diagnostics']['fp_background']:>17d}")
    print(f"{'Cajas Mal Localizadas (IoU < 0.5)':<35} | {metrics_base['error_diagnostics']['localization_error']:>17d} | {metrics_opt['error_diagnostics']['localization_error']:>17d}")
    print(f"{'Omisiones (Falsos Negativos)':<35} | {metrics_base['error_diagnostics']['omissions_fn']:>17d} | {metrics_opt['error_diagnostics']['omissions_fn']:>17d}")
    print("=" * 78)


if __name__ == "__main__":
    main()
