"""Importador de fotos reales de residuos desde Trash Detection Dataset (Downloads).

Mapeo exacto de las 6 clases obligatorias:
  0: BIODEGRADABLE -> organicos
  1: CARDBOARD     -> carton
  2: GLASS         -> vidrio
  3: METAL         -> metal
  4: PAPER         -> papel
  5: PLASTIC       -> plastico

Extrae un conjunto balanceado con anotaciones reales de cajas (bounding boxes)
y lo guarda en data/real_waste/ con partición estricta anti-leakage.
"""
import io
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image

from utils.annotation_storage import (
    AnnotationRecord, AnnotationDataset, CLASSES
)

ZIP_PATH = Path("c:/Users/Anderson Ojeda/Downloads/Trash Detection Dataset.zip")
OUTPUT_DIR = Path("data/real_waste")

CLASS_MAP = {
    0: "organicos",
    1: "carton",
    2: "vidrio",
    3: "metal",
    4: "papel",
    5: "plastico"
}


def import_real_dataset(target_per_class: int = 50, seed: int = 42):
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo zip en: {ZIP_PATH}")

    random.seed(seed)
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / "annotations.json"

    dataset = AnnotationDataset(str(db_path))
    dataset.records.clear()

    print(f"[Importador] Leyendo catálogo desde {ZIP_PATH.name}...")
    with zipfile.ZipFile(str(ZIP_PATH), "r") as z:
        all_files = z.namelist()
        
        # Mapear pares (imagen_path, label_path)
        img_map = {}
        lbl_map = {}
        for f in all_files:
            if f.endswith(".jpg") and "images" in f:
                stem = Path(f).stem
                img_map[stem] = f
            elif f.endswith(".txt") and "labels" in f:
                stem = Path(f).stem
                lbl_map[stem] = f

        common_stems = set(img_map.keys()) & set(lbl_map.keys())
        print(f"[Importador] Total de imágenes anotadas disponibles: {len(common_stems)}")

        # Clasificar por categoría predominante
        stems_by_class = defaultdict(list)
        for stem in common_stems:
            lbl_file = lbl_map[stem]
            txt = z.read(lbl_file).decode("utf-8").strip()
            if not txt:
                stems_by_class["empty"].append(stem)
                continue
            
            cids_in_file = set()
            for line in txt.splitlines():
                parts = line.split()
                if parts:
                    try:
                        cids_in_file.add(int(parts[0]))
                    except ValueError:
                        pass
            
            # Priorizar vidrio y plástico para resolver la confusión
            if 2 in cids_in_file:
                stems_by_class["vidrio"].append(stem)
            elif 5 in cids_in_file:
                stems_by_class["plastico"].append(stem)
            else:
                for cid in cids_in_file:
                    cname = CLASS_MAP.get(cid)
                    if cname:
                        stems_by_class[cname].append(stem)
                        break

        print("\nDistribución disponible de candidatos:")
        for cname in CLASSES:
            print(f"  • {cname:<10}: {len(stems_by_class[cname])} fotos")

        # Seleccionar muestra balanceada
        selected_stems = []
        for cname in CLASSES:
            available = stems_by_class[cname]
            take = min(target_per_class, len(available))
            sample = random.sample(available, take)
            selected_stems.extend(sample)
            print(f"[Selección] {cname}: {take} fotos reales seleccionadas")

        # Añadir 15 fotos vacías (negativas) si existen
        if stems_by_class["empty"]:
            take_empty = min(15, len(stems_by_class["empty"]))
            selected_stems.extend(random.sample(stems_by_class["empty"], take_empty))

        # Mezclar y particionar
        random.shuffle(selected_stems)
        total_selected = len(selected_stems)
        
        print(f"\n[Importador] Extrayendo y convirtiendo {total_selected} fotos reales...")
        extracted_count = 0

        for i, stem in enumerate(selected_stems):
            img_zip_path = img_map[stem]
            lbl_zip_path = lbl_map[stem]

            # Partición estricta
            if i < int(total_selected * 0.70):
                split = "train"
                session_id = f"sess_train_{(i % 4) + 1}"
            elif i < int(total_selected * 0.85):
                split = "val"
                session_id = "sess_val_1"
            else:
                split = "test"
                session_id = "sess_test_1"

            # Extraer y redimensionar imagen a 256x256
            img_bytes = z.read(img_zip_path)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            orig_w, orig_h = pil_img.size

            out_img_name = f"real_{i:04d}_{stem[:16]}.jpg"
            out_img_path = images_dir / out_img_name
            pil_img.resize((256, 256), Image.Resampling.BILINEAR).save(out_img_path, quality=92)

            # Parsear cajas normalizadas YOLO (cx, cy, w, h) -> pixeles en 256x256
            lbl_txt = z.read(lbl_zip_path).decode("utf-8").strip()
            boxes = []
            
            for line in lbl_txt.splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cname = CLASS_MAP.get(cid)
                    if cname:
                        cx = float(parts[1]) * 256.0
                        cy = float(parts[2]) * 256.0
                        bw = float(parts[3]) * 256.0
                        bh = float(parts[4]) * 256.0
                        
                        x1 = max(0.0, cx - bw / 2.0)
                        y1 = max(0.0, cy - bh / 2.0)
                        x2 = min(256.0, cx + bw / 2.0)
                        y2 = min(256.0, cy + bh / 2.0)
                        
                        if (x2 - x1) >= 8.0 and (y2 - y1) >= 8.0:
                            boxes.append({
                                "x1": round(x1, 1),
                                "y1": round(y1, 1),
                                "x2": round(x2, 1),
                                "y2": round(y2, 1),
                                "label": cname
                            })

            dataset.add_or_update(
                image_path=str(out_img_path),
                video_id=f"video_{session_id}",
                session_id=session_id,
                frame_index=i,
                boxes=boxes,
                is_empty_reviewed=(len(boxes) == 0),
                split=split,
                width=256,
                height=256
            )
            extracted_count += 1

        dataset.save()
        print(f"\n[Completado] {extracted_count} imágenes reales guardadas en {OUTPUT_DIR}")
        print(f"[Anotaciones] Registradas en: {db_path}")
        return dataset


if __name__ == "__main__":
    import_real_dataset(target_per_class=60)
