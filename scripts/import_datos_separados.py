"""Importa y unifica el dataset DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1
generando data/datos_separados/annotations.json compatible con el sistema de entrenamiento.
"""
import glob
import json
import os
import random
from pathlib import Path

def build_unified_dataset():
    base = Path("data/DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1")
    if not base.exists():
        alt = Path(r"C:\Users\Anderson Ojeda\Downloads\DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1-20261125T170423Z-1-002\DATOS_SEPARADOS_6_CATEGORIAS_20261124_v1")
        if alt.exists():
            base = alt
        else:
            raise FileNotFoundError("No se encontró la carpeta DATOS_SEPARADOS")

    print(f"[Importador] Leyendo imágenes de: {base}")
    disk_images = {}
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                disk_images[f] = os.path.join(root, f)

    json_files = glob.glob(f"{base}/**/_annotations.coco.json", recursive=True)
    print(f"[Importador] Archivos COCO JSON encontrados: {len(json_files)}")

    cat_normalization = {
        'organicos': 'organicos', 'org-nicos': 'organicos', 'biodegradable': 'organicos',
        'plastico': 'plastico', 'plastic': 'plastico',
        'papel': 'papel', 'paper': 'papel',
        'carton': 'carton', 'cardboard': 'carton',
        'vidrio': 'vidrio', 'glass': 'vidrio',
        'metal': 'metal'
    }

    records_dict = {}

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)

        cat_map = {}
        for c in d.get("categories", []):
            raw = c["name"].lower().replace("ó", "o").replace("á", "a").strip()
            mapped = cat_normalization.get(raw, "desconocido")
            cat_map[c["id"]] = mapped

        img_map = {img["id"]: img for img in d.get("images", []) if img["file_name"] in disk_images}

        for ann in d.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in img_map:
                continue
            img_info = img_map[img_id]
            fname = img_info["file_name"]
            fpath = disk_images[fname]
            cid = ann.get("category_id")
            cname = cat_map.get(cid, "desconocido")
            if cname == "desconocido":
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4 or bbox[2] <= 2 or bbox[3] <= 2:
                continue

            if fname not in records_dict:
                sp = "train"
                if "\\test\\" in fpath or "/test/" in fpath:
                    sp = "test"
                elif "\\val\\" in fpath or "/val/" in fpath:
                    sp = "val"
                records_dict[fname] = {
                    "image_path": str(Path(fpath)),
                    "video_id": f"session_{fname[:12]}",
                    "session_id": f"sess_{fname[:12]}",
                    "frame_index": 0,
                    "boxes": [],
                    "is_empty_reviewed": False,
                    "split": sp,
                    "width": img_info.get("width", 512),
                    "height": img_info.get("height", 512)
                }

            records_dict[fname]["boxes"].append({
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[0] + bbox[2]),
                "y2": float(bbox[1] + bbox[3]),
                "label": cname
            })

    out_dir = Path("data/datos_separados")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "annotations.json"

    dataset_payload = {
        "version": "1.0",
        "description": "Dataset unificado DATOS_SEPARADOS_6_CATEGORIAS",
        "classes": ["plastico", "papel", "carton", "vidrio", "metal", "organicos"],
        "records": list(records_dict.values())
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Dataset unificado guardado exitosamente en: {out_json}")
    print(f"Total imágenes: {len(records_dict)}")
    
    splits = {}
    classes = {}
    for r in records_dict.values():
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        for b in r["boxes"]:
            classes[b["label"]] = classes.get(b["label"], 0) + 1

    print(f"Particiones: {splits}")
    print(f"Cajas por clase: {classes}")

if __name__ == "__main__":
    build_unified_dataset()
