"""Reporte detallado de datos, particiones y objetos anotados.

Informa:
- Cantidad de videos y sesiones de grabación.
- Cantidad de imágenes totales, pendientes y revisadas vacías (negativos).
- Distribución de objetos por clase y partición (train, val, test).
- Detección de posibles desbalances o clases sin suficientes muestras.

Uso:
    python -m scripts.data_summary [--db data/annotations/annotations.json] [--frames_dir data/annotated_frames]
"""
import argparse
import json
import os
from pathlib import Path
from collections import Counter

# Evitar duplicación de OpenMP en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.annotation_storage import AnnotationDataset, CLASSES, CLASS_DISPLAY


def print_data_summary(db_path: str = "data/annotations/annotations.json", frames_dir: str = "data/annotated_frames"):
    dataset = AnnotationDataset(db_path)
    records = list(dataset.records.values())

    # Contabilizar fotogramas en disco para identificar pendientes de anotar
    total_frames_on_disk = 0
    frames_path = Path(frames_dir)
    disk_images = set()
    if frames_path.exists():
        for p in frames_path.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                disk_images.add(str(p.resolve()))
    total_frames_on_disk = len(disk_images)

    annotated_paths = {str(Path(r.image_path).resolve()) for r in records}
    pending_count = len(disk_images - annotated_paths)

    videos = sorted(list({r.video_id for r in records}))
    sessions = sorted(list({r.session_id for r in records}))

    splits = ["train", "val", "test"]
    split_records = {sp: [r for r in records if r.split == sp] for sp in splits}

    empty_reviewed_per_split = {
        sp: sum(1 for r in split_records[sp] if r.is_empty_reviewed and len(r.boxes) == 0)
        for sp in splits
    }
    with_objects_per_split = {
        sp: sum(1 for r in split_records[sp] if len(r.boxes) > 0)
        for sp in splits
    }

    objects_per_split_and_class = {sp: Counter() for sp in splits}
    for sp in splits:
        for r in split_records[sp]:
            for b in r.boxes:
                objects_per_split_and_class[sp][b["label"]] += 1

    print("=" * 78)
    print("RESUMEN GENERAL DEL CONJUNTO DE DATOS - PROYECTO 4 (VISION-IA UCC PASTO)")
    print("=" * 78)
    print(f"Base de datos de anotaciones : {db_path}")
    print(f"Total videos registrados    : {len(videos)} ({', '.join(videos) if videos else 'Ninguno'})")
    print(f"Total sesiones de captura   : {len(sessions)} ({', '.join(sessions) if sessions else 'Ninguna'})")
    print(f"Fotogramas totales en disco : {total_frames_on_disk}")
    print(f"Fotogramas anotados/revisados: {len(records)}")
    print(f"Fotogramas pendientes anot.: {pending_count} {'(ATENCIÓN: hay imágenes sin revisar)' if pending_count > 0 else '(al día)'}")
    print("-" * 78)

    print("\n1. DISTRIBUCIÓN DE FOTOGRAMAS POR PARTICIÓN (ESTRICTA POR SESIÓN)")
    print(f"{'Estado':<28} {'Train':>12} {'Val':>12} {'Test':>12} {'TOTAL':>12}")
    print("-" * 78)
    print(f"{'Con residuos detectados':<28} "
          f"{with_objects_per_split['train']:>12} "
          f"{with_objects_per_split['val']:>12} "
          f"{with_objects_per_split['test']:>12} "
          f"{sum(with_objects_per_split.values()):>12}")
    print(f"{'Revisados vacíos (sin residuo)':<28} "
          f"{empty_reviewed_per_split['train']:>12} "
          f"{empty_reviewed_per_split['val']:>12} "
          f"{empty_reviewed_per_split['test']:>12} "
          f"{sum(empty_reviewed_per_split.values()):>12}")
    print(f"{'Subtotal revisados':<28} "
          f"{len(split_records['train']):>12} "
          f"{len(split_records['val']):>12} "
          f"{len(split_records['test']):>12} "
          f"{len(records):>12}")
    print("-" * 78)

    print("\n2. DISTRIBUCIÓN DE OBJETOS ANOTADOS POR CLASE (CAJAS DELIMITADORAS)")
    print(f"{'Clase':<16} {'Grupo Operativo':<18} {'Train':>10} {'Val':>10} {'Test':>10} {'TOTAL':>10}")
    print("-" * 78)

    total_objs_split = {sp: 0 for sp in splits}
    for cls in CLASSES:
        name = CLASS_DISPLAY.get(cls, cls)
        group = "Orgánico" if cls == "organicos" else "Inorgánico"
        c_train = objects_per_split_and_class["train"][cls]
        c_val = objects_per_split_and_class["val"][cls]
        c_test = objects_per_split_and_class["test"][cls]
        total_cls = c_train + c_val + c_test
        total_objs_split["train"] += c_train
        total_objs_split["val"] += c_val
        total_objs_split["test"] += c_test
        print(f"{name:<16} {group:<18} {c_train:>10} {c_val:>10} {c_test:>10} {total_cls:>10}")

    print("-" * 78)
    print(f"{'TOTAL OBJETOS':<35} "
          f"{total_objs_split['train']:>10} "
          f"{total_objs_split['val']:>10} "
          f"{total_objs_split['test']:>10} "
          f"{sum(total_objs_split.values()):>10}")
    print("=" * 78)

    # Verificación de cobertura
    missing_test = [cls for cls in CLASSES if objects_per_split_and_class["test"][cls] == 0]
    if missing_test:
        print(f"\n[NOTA DE EVALUACIÓN]: Las siguientes clases no tienen muestras en el conjunto test reservado:")
        for m in missing_test:
            print(f"  - {CLASS_DISPLAY[m]}: Se requiere anotar videos con esta clase para evaluar su Recall.")
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Resumen de datos y objetos anotados por clase")
    parser.add_argument("--db", type=str, default="data/annotations/annotations.json", help="Ruta al JSON de anotaciones")
    parser.add_argument("--frames_dir", type=str, default="data/annotated_frames", help="Directorio de fotogramas")
    args = parser.parse_args()

    print_data_summary(args.db, args.frames_dir)


if __name__ == "__main__":
    main()
