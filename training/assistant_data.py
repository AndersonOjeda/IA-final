"""Dataset enriquecido de intenciones y consultas en español para el Asistente de Residuos.

Cubre exhaustivamente las 6 intenciones y las entidades de material:
1. resumen_deteccion (¿Qué residuos ves?)
2. separar_especifico (¿Cómo separo esta lata/botella/caja?)
3. consultar_organico (¿Hay algo orgánico/comida/cáscaras?)
4. condicion_suciedad (¿Qué hago si está sucio/grasoso/mojado?)
5. donde_botar_ambiguo (¿Dónde lo boto? / ¿En qué caneca va?)
6. fuera_de_alcance (Saludos, chistes, economía, matemáticas, clima, política, etc.)
"""
from typing import List, Tuple

INTENTS = [
    "resumen_deteccion",
    "separar_especifico",
    "consultar_organico",
    "condicion_suciedad",
    "donde_botar_ambiguo",
    "fuera_de_alcance"
]

INTENT_TO_ID = {name: idx for idx, name in enumerate(INTENTS)}
ID_TO_INTENT = {idx: name for idx, name in enumerate(INTENTS)}

TRAIN_DATA: List[Tuple[str, str, str]] = [
    # ---------------- 1. Resumen de detecciones ----------------
    ("¿Qué residuos ves en la mesa?", "resumen_deteccion", "none"),
    ("¿Qué objetos has detectado?", "resumen_deteccion", "none"),
    ("¿Qué basura hay en la cámara?", "resumen_deteccion", "none"),
    ("Dime qué residuos están presentes", "resumen_deteccion", "none"),
    ("¿Qué estás viendo ahora mismo?", "resumen_deteccion", "none"),
    ("Muéstrame los residuos identificados", "resumen_deteccion", "none"),
    ("¿Cuáles son los objetos en pantalla?", "resumen_deteccion", "none"),
    ("¿Qué residuos hay en la escena?", "resumen_deteccion", "none"),
    ("Enumera los desechos que detectas", "resumen_deteccion", "none"),
    ("¿Puedes decirme qué basura ves?", "resumen_deteccion", "none"),
    ("¿Qué materiales lograste reconocer?", "resumen_deteccion", "none"),
    ("Hazme un resumen de los objetos visibles", "resumen_deteccion", "none"),
    ("¿Hay basura identificada en la imagen?", "resumen_deteccion", "none"),
    ("Lista de residuos encontrados por favor", "resumen_deteccion", "none"),
    ("¿Qué elementos hay detectados?", "resumen_deteccion", "none"),

    # ---------------- 2. Separación específica ----------------
    ("¿Cómo separo esta lata?", "separar_especifico", "metal"),
    ("¿Dónde debo botar esta botella de plástico?", "separar_especifico", "plastico"),
    ("¿En qué caneca van las botellas de refresco?", "separar_especifico", "plastico"),
    ("¿En qué caneca pongo las botellas de gaseosa?", "separar_especifico", "plastico"),
    ("¿Cómo se recicla el cartón?", "separar_especifico", "carton"),
    ("¿Qué hago con este frasco de vidrio?", "separar_especifico", "vidrio"),
    ("¿Cómo separo las hojas de papel?", "separar_especifico", "papel"),
    ("¿Cómo debo disponer la lata de gaseosa?", "separar_especifico", "metal"),
    ("¿En qué bolsa meto el envase de plástico?", "separar_especifico", "plastico"),
    ("¿Cómo reciclo una caja de cartón?", "separar_especifico", "carton"),
    ("¿Qué hago con una botella de vidrio vacía?", "separar_especifico", "vidrio"),
    ("¿Cómo desecho el papel periódico?", "separar_especifico", "papel"),
    ("¿Cómo separo estos restos de comida?", "separar_especifico", "organicos"),
    ("¿Dónde coloco la cáscara de plátano?", "separar_especifico", "organicos"),
    ("¿Dónde tiro los restos de manzana?", "separar_especifico", "organicos"),
    ("¿En qué caneca va la lata de atún?", "separar_especifico", "metal"),
    ("¿Dónde deposito el cartón corrugado?", "separar_especifico", "carton"),
    ("¿Qué hago con un vaso de plástico desechable?", "separar_especifico", "plastico"),
    ("¿Cómo separar botellas de vidrio de cerveza?", "separar_especifico", "vidrio"),
    ("¿Dónde se botan las hojas de cuaderno?", "separar_especifico", "papel"),

    # ---------------- 3. Consulta de orgánicos ----------------
    ("¿Hay algo orgánico en la escena?", "consultar_organico", "organicos"),
    ("¿Ves algún residuo orgánico?", "consultar_organico", "organicos"),
    ("¿Hay restos de comida frente a la cámara?", "consultar_organico", "organicos"),
    ("¿Detectas algún material compostable?", "consultar_organico", "organicos"),
    ("¿Aparece algún desecho orgánico?", "consultar_organico", "organicos"),
    ("¿Hay fruta o comida en la imagen?", "consultar_organico", "organicos"),
    ("¿Se encuentra algún residuo de origen orgánico?", "consultar_organico", "organicos"),
    ("Dime si hay restos orgánicos", "consultar_organico", "organicos"),
    ("¿Ves restos de alimentos o cáscaras?", "consultar_organico", "organicos"),
    ("¿Identificas desechos de cocina?", "consultar_organico", "organicos"),
    ("¿Hay algún residuo para compostaje?", "consultar_organico", "organicos"),
    ("¿Hay materia orgánica visible?", "consultar_organico", "organicos"),

    # ---------------- 4. Condición de suciedad / contaminación ----------------
    ("¿Qué hago si el cartón está sucio?", "condicion_suciedad", "carton"),
    ("¿Qué pasa si la caja tiene grasa de pizza?", "condicion_suciedad", "carton"),
    ("¿Cómo separo el papel si está manchado de comida?", "condicion_suciedad", "papel"),
    ("¿Se puede reciclar una botella plástica con líquidos adentro?", "condicion_suciedad", "plastico"),
    ("¿Qué hago si el envase de metal está sucio y con grasa?", "condicion_suciedad", "metal"),
    ("¿El papel con aceite va en la caneca blanca?", "condicion_suciedad", "papel"),
    ("¿Qué hago si el cartón está mojado o húmedo?", "condicion_suciedad", "carton"),
    ("¿Puedo reciclar un frasco con restos de mermelada?", "condicion_suciedad", "vidrio"),
    ("¿Qué pasa si la servilleta está untada de comida?", "condicion_suciedad", "papel"),
    ("¿El plástico con aceite automotriz se puede reciclar?", "condicion_suciedad", "plastico"),
    ("¿Qué hacer con una lata llena de moho o restos podridos?", "condicion_suciedad", "metal"),
    ("¿El papel higiénico sucio se recicla?", "condicion_suciedad", "papel"),
    ("¿Cómo dispongo una caja de cartón impregnada de aceite?", "condicion_suciedad", "carton"),
    ("¿Se puede aprovechar una botella de vidrio con residuos de salsa?", "condicion_suciedad", "vidrio"),

    # ---------------- 5. Consulta ambigua de dónde botar ----------------
    ("¿Dónde lo boto?", "donde_botar_ambiguo", "none"),
    ("¿En qué caneca va esto?", "donde_botar_ambiguo", "none"),
    ("¿En qué bolsa lo pongo?", "donde_botar_ambiguo", "none"),
    ("¿Dónde se desecha?", "donde_botar_ambiguo", "none"),
    ("¿En qué color de bolsa se tira?", "donde_botar_ambiguo", "none"),
    ("¿Dónde deposito este residuo?", "donde_botar_ambiguo", "none"),
    ("¿A qué caneca corresponde?", "donde_botar_ambiguo", "none"),
    ("¿En cuál caneca tengo que tirarlo?", "donde_botar_ambiguo", "none"),
    ("¿Dónde va a parar esto?", "donde_botar_ambiguo", "none"),
    ("¿En cuál caneca lo coloco?", "donde_botar_ambiguo", "none"),
    ("¿Dónde debo arrojar esto?", "donde_botar_ambiguo", "none"),
    ("¿En qué caneca lo echo?", "donde_botar_ambiguo", "none"),
    ("¿Dónde debo tirarlo?", "donde_botar_ambiguo", "none"),

    # ---------------- 6. Consultas fuera de alcance ----------------
    ("¿Cuál es el clima de hoy en Pasto?", "fuera_de_alcance", "none"),
    ("¿Quién ganó el partido de fútbol?", "fuera_de_alcance", "none"),
    ("Explícame cómo resolver una ecuación diferencial", "fuera_de_alcance", "none"),
    ("¿Qué hora es en este momento?", "fuera_de_alcance", "none"),
    ("Escribe una poesía sobre el mar", "fuera_de_alcance", "none"),
    ("¿Cuánto cuesta un pasaje de avión a Bogotá?", "fuera_de_alcance", "none"),
    ("Recomiéndame una película de ciencia ficción", "fuera_de_alcance", "none"),
    ("¿Quién es el docente Cristian Ordoñez?", "fuera_de_alcance", "none"),
    ("Hola, buenos días", "fuera_de_alcance", "none"),
    ("Buenas tardes, ¿cómo estás?", "fuera_de_alcance", "none"),
    ("Cuéntame un chiste gracioso", "fuera_de_alcance", "none"),
    ("Dime una broma o chiste", "fuera_de_alcance", "none"),
    ("¿A cómo está el dólar hoy en Colombia?", "fuera_de_alcance", "none"),
    ("¿Cuál es el precio del dólar?", "fuera_de_alcance", "none"),
    ("¿Cómo hago una base de datos en SQL?", "fuera_de_alcance", "none"),
    ("¿Cuál es la capital de Nariño?", "fuera_de_alcance", "none"),
    ("Háblame sobre la historia de Colombia", "fuera_de_alcance", "none"),
    ("¿Quién es el presidente de Colombia?", "fuera_de_alcance", "none"),
    ("¿Qué música me recomiendas escuchar?", "fuera_de_alcance", "none"),
    ("¿Qué opinas de la inteligencia artificial?", "fuera_de_alcance", "none")
]

# Conjunto de prueba independiente (desacoplado y exigente)
TEST_DATA: List[Tuple[str, str, str]] = [
    ("¿Qué tipo de basura estás detectando?", "resumen_deteccion", "none"),
    ("¿Cuáles elementos reconoces ahora?", "resumen_deteccion", "none"),
    ("¿Cómo tengo que desechar esta lata de atún?", "separar_especifico", "metal"),
    ("¿En qué caneca pongo las botellas de refresco?", "separar_especifico", "plastico"),
    ("¿Cómo reciclo estas cajas de cartón corrugado?", "separar_especifico", "carton"),
    ("¿Dónde va el frasco de café?", "separar_especifico", "vidrio"),
    ("¿Hay algún desecho de alimentos sobre la mesa?", "consultar_organico", "organicos"),
    ("¿Aparece comida en la cámara?", "consultar_organico", "organicos"),
    ("¿Qué debo hacer si la servilleta está llena de grasa?", "condicion_suciedad", "papel"),
    ("¿Y si la caja de pizza tiene restos de queso?", "condicion_suciedad", "carton"),
    ("¿En qué caneca lo echo?", "donde_botar_ambiguo", "none"),
    ("¿Dónde debo tirarlo?", "donde_botar_ambiguo", "none"),
    ("¿Cómo está la cotización del dólar hoy?", "fuera_de_alcance", "none"),
    ("Cuéntame un chiste corto", "fuera_de_alcance", "none")
]
