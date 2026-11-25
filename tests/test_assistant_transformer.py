"""Pruebas unitarias para el Transformer del Asistente y el Motor Normativo."""
import os
import unittest
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from training.assistant_transformer import (
    SimpleSpanishTokenizer, WasteAssistantTransformer,
    ColombianRecyclingKnowledgeEngine
)


class AssistantTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_path = Path("models/runtime/assistant_vocab.json")
        cls.weights_path = Path("models/runtime/assistant_transformer.pth")

        cls.tokenizer = SimpleSpanishTokenizer()
        if cls.vocab_path.exists():
            cls.tokenizer.load(str(cls.vocab_path))
        else:
            cls.tokenizer.build_vocab(["¿dónde lo boto?", "¿qué residuos ves?", "lata de metal"])

        cls.engine = ColombianRecyclingKnowledgeEngine()

        if cls.weights_path.exists():
            payload = torch.load(cls.weights_path, map_location="cpu", weights_only=True)
            cls.model = WasteAssistantTransformer(
                vocab_size=payload["vocab_size"],
                d_model=payload["d_model"],
                nhead=payload["nhead"],
                num_layers=payload["num_layers"]
            )
            cls.model.load_state_dict(payload["model_state_dict"])
            cls.model.eval()
        else:
            cls.model = None

    def test_ambiguity_multiple_objects(self):
        """Verifica que si hay múltiples objetos en escena y se pregunta '¿Dónde lo boto?',
        el motor solicite aclaración explícita."""
        scene = [
            {"class_name": "plastico", "confidence": 0.90},
            {"class_name": "carton", "confidence": 0.85}
        ]
        resp = self.engine.answer_query(
            intent="donde_botar_ambiguo",
            material_entity="none",
            scene_objects=scene
        )
        self.assertEqual(resp["tipo"], "ambiguedad_multiples_residuos")
        self.assertIn("ambigua", resp["respuesta"].lower())
        self.assertIn("Plástico", resp["respuesta"])
        self.assertIn("Cartón", resp["respuesta"])

    def test_single_object_inferred(self):
        """Verifica que con un solo objeto en escena, la pregunta ambigua se resuelva directamente."""
        scene = [{"class_name": "vidrio", "confidence": 0.92}]
        resp = self.engine.answer_query(
            intent="donde_botar_ambiguo",
            material_entity="none",
            scene_objects=scene
        )
        self.assertEqual(resp["tipo"], "separacion_inferida")
        self.assertIn("Blanca", resp["respuesta"])
        self.assertIn("Vidrio", resp["respuesta"])

    def test_dirty_or_contaminated_condition(self):
        """Verifica la respuesta condicional para materiales sucios / contaminados."""
        resp = self.engine.answer_query(
            intent="condicion_suciedad",
            material_entity="carton",
            scene_objects=[]
        )
        self.assertEqual(resp["tipo"], "condicion_suciedad")
        self.assertIn("Negra", resp["respuesta"])
        self.assertIn("Blanca", resp["respuesta"])

    def test_out_of_scope_query(self):
        """Verifica el rechazo amable de preguntas fuera de alcance."""
        resp = self.engine.answer_query(
            intent="fuera_de_alcance",
            material_entity="none",
            scene_objects=[]
        )
        self.assertEqual(resp["tipo"], "fuera_de_alcance")
        self.assertIn("Asistente Experto en Clasificación de Residuos", resp["respuesta"])


if __name__ == "__main__":
    unittest.main()
