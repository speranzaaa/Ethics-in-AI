"""LLM evaluation layer - requires a GPU !!!"""

import json
import logging
import warnings

from unsloth import FastLanguageModel

from tqdm.auto import tqdm

from .config import LLMMODEL_NAME, RULE_PATH
from .utils.text import estrai_tupla, CheckRegolaOutput


# Prompt

SYSTEM_PROMPT = """Sei un assistente per il supporto decisionale clinico in ambito pediatrico. Il tuo compito è valutare se, dati i campi clinici di un caso, una specifica regola che descrive un indicatore di possibile abuso o maltrattamento sui minori risulta VIOLATA.

Definizione di "regola violata":
- La regola è VIOLATA quando la situazione descritta nella regola è esplicitamente presente o chiaramente compatibile con quanto riportato nei dati clinici forniti.
- Se i dati clinici non contengono informazioni che supportano la regola, la regola NON è violata.

Criteri:
- Non inferire né ipotizzare situazioni non supportate dal testo dei dati clinici.
- Sii conservativo: in caso di dubbio o evidenza debole, la regola NON è violata.
- Considera l'insieme di tutti i campi clinici forniti, non un campo alla volta.

Devi rispondere ESCLUSIVAMENTE con una tupla, senza testo aggiuntivo, senza markdown, senza commenti.
Il primo valore si riferisce alla violazione della regola, True se è violata o False altrimenti.
Il secondo campo si riferisce al grado di confidenza della violazione che può andare da 0 a 1.
Esempio di regola violata: (True, 'valore tra 0 e 1')
Esempio di regola non violata: (False, 0) """


def crea_prompt(regola, dati):
    sezione_dati = "DATI CLINICI DEL CASO:\n"
    sezione_dati += f"età: {dati['eta_in_anni']}\n"
    sezione_dati += f"sesso: {dati['sesso']}\n"
    sezione_dati += f"problema_principale: {dati['problema_principale']}\n"
    sezione_dati += f"dati_riferiti: {dati['dati_riferiti']}\n"
    sezione_dati += f"diagnosi: {dati['diagnosi']}\n"
    sezione_dati += f"causale: {dati['causale']}\n"
    sezione_dati += f"anamnesi: {dati['anamnesi']}\n"
    sezione_dati += f"note_aggiuntive: {dati['note_aggiuntive']}\n"

    sezione_regola = f"\nREGOLA DA VALUTARE:\n{regola['descrizione']}\n"
    istruzione = "\nRestituisci SOLO la tupla. "
    return sezione_dati + sezione_regola + istruzione


# Model loading

model = None
tokenizer = None


def load_llm():

    global model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLMMODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    model.generation_config.max_length = None

    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"transformers\.modeling_attn_mask_utils",
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def evaluate_prompt(user_prompt, max_new_tokens=20):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_text = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    )
    return response_text


def valuta_caso(rules, prompt_data):
    violated_rules = []
    for regola in tqdm(rules, desc="Rules evaluation"):
        try:
            rule = {"id": "", "grav": 0, "conf": 0}
            user_prompt = crea_prompt(regola, prompt_data)
            response = evaluate_prompt(user_prompt)
            tupla = estrai_tupla(response)
            parsed = CheckRegolaOutput(ris=tupla[0], conf=tupla[1])
            if parsed.ris:
                rule["id"]   = regola["id"]
                rule["grav"] = regola["gravità"]
                rule["conf"] = parsed.conf
                violated_rules.append(rule)
        except Exception as e:
            print("Errore : ", str(e))
    return violated_rules


def load_rules(rule_path=None):
    if rule_path is None:
        rule_path = RULE_PATH
    with open(rule_path, "r") as f:
        rules_file = json.load(f)
    return rules_file["regole"], rules_file["definizioni"]
