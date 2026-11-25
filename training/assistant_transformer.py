"""Transformer del asistente de residuos en español entrenado desde cero.

Componentes:
1. Tokenizador y vocabulario propio construido a partir de los datos de entrenamiento.
2. Embeddings propios aprendidos desde cero (sin pesos externos).
3. Transformer Encoder en PyTorch (capas de auto-atención multi-cabeza y feed-forward).
4. Cabezas de clasificación para intención y material objetivo mencionado.
5. Motor de decisión y tabla normativa de reciclaje (Resolución 2184 de 2019 de Colombia).
"""
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.assistant_data import INTENTS, INTENT_TO_ID, ID_TO_INTENT, TRAIN_DATA, TEST_DATA
from utils.annotation_storage import CLASSES, CLASS_DISPLAY

MATERIALS = ["none"] + CLASSES
MAT_TO_ID = {m: i for i, m in enumerate(MATERIALS)}
ID_TO_MAT = {i: m for i, m in enumerate(MATERIALS)}


class SimpleSpanishTokenizer:
    """Tokenizador léxico en español con construcción de vocabulario desde cero."""
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"

    def __init__(self):
        self.vocab: Dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.CLS_TOKEN: 2,
            self.SEP_TOKEN: 3
        }
        self.inverse_vocab: Dict[int, str] = {idx: tok for tok, idx in self.vocab.items()}

    def tokenize_text(self, text: str) -> List[str]:
        # Minúsculas y separación de signos de puntuación
        text = text.lower().strip()
        tokens = re.findall(r"\w+|[¿?¡!.,;:\"()]", text, re.UNICODE)
        return tokens

    def build_vocab(self, texts: List[str]):
        for text in texts:
            for tok in self.tokenize_text(text):
                if tok not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                    self.inverse_vocab[idx] = tok

    def encode(self, text: str, max_len: int = 24) -> Tuple[List[int], List[int]]:
        tokens = [self.CLS_TOKEN] + self.tokenize_text(text) + [self.SEP_TOKEN]
        ids = [self.vocab.get(tok, self.vocab[self.UNK_TOKEN]) for tok in tokens]

        # Truncar o rellenar con <pad>
        if len(ids) > max_len:
            ids = ids[:max_len]
            mask = [1] * max_len
        else:
            pad_len = max_len - len(ids)
            mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [self.vocab[self.PAD_TOKEN]] * pad_len

        return ids, mask

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.vocab = data["vocab"]
            self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}


class WasteAssistantTransformer(nn.Module):
    """Transformer Encoder compacto entrenado desde cero para intención y extracción de entidades."""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 32,
        num_intents: int = len(INTENTS),
        num_materials: int = len(MATERIALS)
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cabeza de clasificación de intención
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_intents)
        )

        # Cabeza de extracción de material
        self.material_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_materials)
        )

        self._init_weights()

    def _init_weights(self):
        """Inicialización aleatoria propia (Xavier Uniform) sin pesos externos."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = input_ids.shape
        positions = torch.arange(0, S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model) + self.pos_embedding(positions)

        # Máscara de padding de PyTorch: True en posiciones a ignorar (pad)
        padding_mask = (attention_mask == 0)
        features = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # Usar la representación del token [CLS] (posición 0)
        cls_rep = features[:, 0, :]

        intent_logits = self.intent_head(cls_rep)
        material_logits = self.material_head(cls_rep)
        return intent_logits, material_logits


class ColombianRecyclingKnowledgeEngine:
    """Motor de reglas normativas para la separación de residuos en Colombia (Resolución 2184 de 2019).

    Distingue:
    - Lo que el modelo detecta (material / clase visual).
    - Lo que la cámara no puede asegurar (limpieza, contaminación interna, grasa).
    - Ambigüedad en preguntas de dónde botar cuando hay múltiples objetos en la escena.
    """
    NORM_RULES = {
        "plastico": {
            "nombre": "Plástico",
            "grupo": "Inorgánico",
            "caneca_limpio": "Blanca (Residuos aprovechables)",
            "caneca_sucio": "Negra (No aprovechables) si no es posible limpiarlo",
            "consejo": "Vacía el envase, enjuágalo para retirar restos de producto y déjalo secar. Si está limpio y seco, va a la bolsa blanca."
        },
        "papel": {
            "nombre": "Papel",
            "grupo": "Inorgánico",
            "caneca_limpio": "Blanca (Residuos aprovechables)",
            "caneca_sucio": "Negra (Residuos no aprovechables)",
            "consejo": "Debe estar limpio, seco y no arrugado en exceso. El papel sucio, encerado, plastificado, con grasa o papel higiénico va obligatoriamente a la caneca negra."
        },
        "carton": {
            "nombre": "Cartón",
            "grupo": "Inorgánico",
            "caneca_limpio": "Blanca (Residuos aprovechables)",
            "caneca_sucio": "Negra (Residuos no aprovechables)",
            "consejo": "Desarma y aplana las cajas para optimizar espacio. Si la caja tiene grasa (como cajas de pizza con queso o aceite), esa sección no se puede reciclar y va a la caneca negra."
        },
        "vidrio": {
            "nombre": "Vidrio",
            "grupo": "Inorgánico",
            "caneca_limpio": "Blanca (Residuos aprovechables)",
            "caneca_sucio": "Blanca (tras vaciar y enjuagar)",
            "consejo": "Botellas y frascos enteros van limpios y secos a la caneca blanca. Si está roto, envuélvelo en papel grueso o cartón para proteger la seguridad del reciclador."
        },
        "metal": {
            "nombre": "Metal",
            "grupo": "Inorgánico",
            "caneca_limpio": "Blanca (Residuos aprovechables)",
            "caneca_sucio": "Blanca (tras escurrir)",
            "consejo": "Latas de aluminio y conservas deben estar escurridas y limpias en la caneca blanca. No aplastes los bordes filosos para evitar cortaduras."
        },
        "organicos": {
            "nombre": "Restos Orgánicos",
            "grupo": "Orgánico",
            "caneca_limpio": "Verde (Residuos orgánicos aprovechables)",
            "caneca_sucio": "Verde (Residuos orgánicos aprovechables)",
            "consejo": "Restos de comida, cáscaras de frutas, vegetales y desechos agrícolas van a la caneca verde para compostaje y abono orgánico. No incluir bolsas plásticas ni servilletas."
        }
    }

    def answer_query(
        self,
        intent: str,
        material_entity: str,
        scene_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce la respuesta contextual uniendo la intención del Transformer con la escena visual."""
        detected_classes = list({obj.get("class_name") for obj in scene_objects if obj.get("class_name") in CLASSES})

        if intent == "fuera_de_alcance":
            return {
                "respuesta": "Soy el Asistente Experto en Clasificación de Residuos de la UCC Campus Pasto. Solo puedo ayudarte con preguntas sobre residuos detectados en cámara, normas de reciclaje y el Código de Colores de Colombia (Resolución 2184 de 2019).",
                "tipo": "fuera_de_alcance"
            }

        if intent == "resumen_deteccion":
            if not scene_objects:
                return {
                    "respuesta": "En este momento no detecto ningún residuo con suficiente confianza en la escena. Asegúrate de enfocar bien el objeto ante la cámara.",
                    "tipo": "resumen_vacio"
                }
            names = [CLASS_DISPLAY.get(c, c) for c in detected_classes]
            count_str = ", ".join(f"{sum(1 for o in scene_objects if o.get('class_name') == c)} objeto(s) de {CLASS_DISPLAY[c]}" for c in detected_classes)
            return {
                "respuesta": f"En la escena actual he localizado {len(scene_objects)} residuo(s): {count_str}. Según el código nacional, debes clasificarlos entre caneca blanca (aprovechables) y caneca verde (orgánicos).",
                "tipo": "resumen",
                "clases_detectadas": names
            }

        if intent == "consultar_organico":
            has_organic = "organicos" in detected_classes
            if has_organic:
                return {
                    "respuesta": "Sí, he detectado restos orgánicos en la escena. Debes depositarlos en la Caneca Verde para aprovechamiento en compostaje.",
                    "tipo": "organico_presente"
                }
            else:
                return {
                    "respuesta": "No detecto residuos orgánicos en este momento. Los objetos observados pertenecen al grupo inorgánico/aprovechable.",
                    "tipo": "organico_ausente"
                }

        if intent == "condicion_suciedad":
            target_mat = material_entity if material_entity != "none" else (detected_classes[0] if detected_classes else "carton")
            rule = self.NORM_RULES.get(target_mat, self.NORM_RULES["carton"])
            return {
                "respuesta": (
                    f"Atención a la condición de {rule['nombre']}: La cámara identifica el tipo de material, pero tú debes verificar su estado. "
                    f"Si el residuo está LIMPIO Y SECO, su destino es la Caneca {rule['caneca_limpio']}. "
                    f"Sin embargo, si está SUCIO, con grasa, comida o humedad contaminante, su destino cambia a la Caneca {rule['caneca_sucio']}."
                ),
                "tipo": "condicion_suciedad",
                "material": rule["nombre"]
            }

        if intent in ("separar_especifico", "donde_botar_ambiguo"):
            # Si el usuario especificó el material en su pregunta (ej: "esta lata"):
            if material_entity in CLASSES:
                rule = self.NORM_RULES[material_entity]
                return {
                    "respuesta": f"Para el residuo de {rule['nombre']} ({rule['grupo']}): {rule['consejo']} Deposítalo en la Caneca {rule['caneca_limpio']}.",
                    "tipo": "separacion_especifica",
                    "material": rule["nombre"]
                }

            # Si la pregunta es ambigua ("¿Dónde lo boto?"):
            if not scene_objects:
                return {
                    "respuesta": "¿Dónde lo boto?: Actualmente no veo ningún objeto claro frente a la cámara. Por favor muestra el residuo a clasificar.",
                    "tipo": "ambiguo_sin_objetos"
                }
            elif len(detected_classes) == 1:
                mat = detected_classes[0]
                rule = self.NORM_RULES[mat]
                return {
                    "respuesta": f"Detecto un residuo de {rule['nombre']} ({rule['grupo']}): {rule['consejo']} Destino: Caneca {rule['caneca_limpio']}.",
                    "tipo": "separacion_inferida",
                    "material": rule["nombre"]
                }
            else:
                # Múltiples clases en la misma escena -> Manejo explícito de ambigüedad
                names = [CLASS_DISPLAY.get(c, c) for c in detected_classes]
                return {
                    "respuesta": f"Pregunta ambigua: Veo varios tipos de residuos en la escena ({', '.join(names)}). ¿A cuál de ellos te refieres? Pregúntame por ejemplo: '¿Dónde boto el {names[0]}?'",
                    "tipo": "ambiguedad_multiples_residuos",
                    "opciones": names
                }

        # Fallback
        return {
            "respuesta": "Para separar correctamente según la Resolución 2184 de Colombia: Blanco para aprovechables limpios y secos, Verde para orgánicos aprovechables, y Negro para no aprovechables.",
            "tipo": "general"
        }
