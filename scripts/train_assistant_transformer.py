"""Entrenamiento y evaluación del Transformer del Asistente en Español desde cero.

Cumple estrictamente:
- Sin pesos ni embeddings preentrenados (entrenados desde cero en PyTorch).
- Vocabulario y tokenización generados a partir de los datos del proyecto.
- Clasificación de intenciones y entidades de residuos comunes en español.
- Evaluación en conjunto de prueba independiente redactado de forma desacoplada.
- Guarda el checkpoint y el vocabulario en models/runtime/.

Uso:
    python -m scripts.train_assistant_transformer [--epochs 35] [--lr 0.001]
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from training.assistant_data import (
    INTENTS, INTENT_TO_ID, ID_TO_INTENT, TRAIN_DATA, TEST_DATA
)
from training.assistant_transformer import (
    SimpleSpanishTokenizer, WasteAssistantTransformer,
    ColombianRecyclingKnowledgeEngine, MATERIALS, MAT_TO_ID, ID_TO_MAT
)


class TextIntentDataset(Dataset):
    def __init__(self, data, tokenizer: SimpleSpanishTokenizer, max_len: int = 24):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, intent, material = self.data[idx]
        input_ids, mask = self.tokenizer.encode(text, self.max_len)
        intent_id = INTENT_TO_ID.get(intent, 0)
        mat_id = MAT_TO_ID.get(material, 0)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(intent_id, dtype=torch.long),
            torch.tensor(mat_id, dtype=torch.long),
            text
        )


def train_assistant(
    epochs: int = 35,
    lr: float = 0.001,
    batch_size: int = 8,
    out_dir: str = "models/runtime",
    seed: int = 42
):
    random.seed(seed)
    torch.manual_seed(seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Construir Tokenizador y Vocabulario desde cero
    print("[Transformer Asistente] Construyendo vocabulario desde cero a partir de frases de entrenamiento...")
    tokenizer = SimpleSpanishTokenizer()
    all_train_texts = [item[0] for item in TRAIN_DATA]
    tokenizer.build_vocab(all_train_texts)

    vocab_file = out_path / "assistant_vocab.json"
    tokenizer.save(str(vocab_file))
    print(f"[Transformer Asistente] Vocabulario construido con {len(tokenizer.vocab)} tokens únicos.")

    # 2. Datasets y DataLoaders
    train_dataset = TextIntentDataset(TRAIN_DATA, tokenizer, max_len=24)
    test_dataset = TextIntentDataset(TEST_DATA, tokenizer, max_len=24)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Inicializar Transformer desde cero
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WasteAssistantTransformer(
        vocab_size=len(tokenizer.vocab),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_len=32,
        num_intents=len(INTENTS),
        num_materials=len(MATERIALS)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_intent_fn = nn.CrossEntropyLoss()
    loss_mat_fn = nn.CrossEntropyLoss()

    print(f"[Transformer Asistente] Arquitectura: 2 capas TransformerEncoder | Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Transformer Asistente] Dispositivo de cómputo: {device}")

    # 4. Bucle de Entrenamiento
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        correct_intents = 0
        total_samples = 0

        for input_ids, mask, intent_id, mat_id, _ in train_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            intent_id = intent_id.to(device)
            mat_id = mat_id.to(device)

            optimizer.zero_grad()
            intent_logits, mat_logits = model(input_ids, mask)

            loss_intent = loss_intent_fn(intent_logits, intent_id)
            loss_mat = loss_mat_fn(mat_logits, mat_id)
            loss = loss_intent + 0.8 * loss_mat

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(input_ids)
            preds = intent_logits.argmax(dim=-1)
            correct_intents += (preds == intent_id).sum().item()
            total_samples += len(input_ids)

        ep_loss = total_loss / total_samples
        ep_acc = correct_intents / total_samples
        if ep % 5 == 0 or ep == 1:
            print(f"Época [{ep:02d}/{epochs:02d}] - Loss: {ep_loss:.4f} | Precisión Train Intención: {ep_acc * 100:.1f}%")

    checkpoint_path = out_path / "assistant_transformer.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": len(tokenizer.vocab),
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "intents": INTENTS,
        "materials": MATERIALS,
        "pretrained": False,
        "seed": seed
    }, checkpoint_path)
    print(f"\n[Transformer Asistente] Checkpoint guardado en: {checkpoint_path}")

    # 5. Evaluación en Conjunto de Prueba Independiente (Test Set)
    print("\n" + "=" * 65)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA INDEPENDIENTE (FRASES NUNCA VISTAS)")
    print("=" * 65)

    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_correct = 0
    per_intent_correct = {i: 0 for i in INTENTS}
    per_intent_total = {i: 0 for i in INTENTS}

    knowledge_engine = ColombianRecyclingKnowledgeEngine()

    print(f"{'Pregunta Test':<45} {'Intención Pred':<22} {'OK':>4}")
    print("-" * 75)
    with torch.no_grad():
        for input_ids, mask, intent_id, mat_id, text in test_loader:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            intent_logits, mat_logits = model(input_ids, mask)

            pred_intent_id = intent_logits.argmax(dim=-1).item()
            pred_mat_id = mat_logits.argmax(dim=-1).item()

            pred_intent = ID_TO_INTENT[pred_intent_id]
            pred_mat = ID_TO_MAT[pred_mat_id]
            gt_intent = ID_TO_INTENT[intent_id.item()]

            is_correct = (pred_intent == gt_intent)
            if is_correct:
                test_correct += 1
                per_intent_correct[gt_intent] += 1
            per_intent_total[gt_intent] += 1

            disp_text = (text[0][:42] + "...") if len(text[0]) > 45 else text[0]
            print(f"{disp_text:<45} {pred_intent:<22} {'SÍ' if is_correct else 'NO':>4}")

    test_acc = test_correct / len(test_dataset)
    print("-" * 75)
    print(f"EXACTITUD EN PRUEBA INDEPENDIENTE (TEST ACCURACY): {test_acc * 100:.2f}%\n")

    # Guardar métricas de evaluación
    eval_summary = {
        "test_accuracy": round(test_acc, 4),
        "total_test_samples": len(test_dataset),
        "correct_predictions": test_correct,
        "per_intent_metrics": {
            i: {
                "total": per_intent_total[i],
                "correct": per_intent_correct[i],
                "accuracy": round(per_intent_correct[i] / per_intent_total[i], 4) if per_intent_total[i] > 0 else 1.0
            }
            for i in INTENTS
        }
    }
    with open(out_path / "assistant_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    # 6. Demostración de consulta y ambigüedad
    print("=" * 65)
    print("DEMOSTRACIÓN DE CONSULTA AMBIGUA CON MÚLTIPLES RESIDUOS DETECTADOS")
    print("=" * 65)
    sample_q = "¿Dónde lo boto?"
    ids, mask = tokenizer.encode(sample_q)
    t_ids = torch.tensor([ids], device=device)
    t_mask = torch.tensor([mask], device=device)
    with torch.no_grad():
        i_log, m_log = model(t_ids, t_mask)
        p_intent = ID_TO_INTENT[i_log.argmax().item()]
        p_mat = ID_TO_MAT[m_log.argmax().item()]

    # Escena con 2 objetos: plástico y vidrio
    fake_scene = [
        {"class_name": "plastico", "confidence": 0.88},
        {"class_name": "vidrio", "confidence": 0.79}
    ]
    resp = knowledge_engine.answer_query(p_intent, p_mat, fake_scene)
    print(f"Consulta: '{sample_q}'")
    print(f"Intención predicha: {p_intent}")
    print(f"Escena detectada: Plástico y Vidrio")
    print(f"Respuesta del Asistente:\n  -> {resp['respuesta']}\n")


def main():
    parser = argparse.ArgumentParser(description="Entrenar Transformer del Asistente")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", type=str, default="models/runtime")
    args = parser.parse_args()

    train_assistant(epochs=args.epochs, lr=args.lr, out_dir=args.out)


if __name__ == "__main__":
    main()
