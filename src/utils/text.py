import ast
import re

from pydantic import BaseModel


def estrai_tupla(testo):
    testo = testo.strip()

    # Rimuove i blocchi di codice markdown
    testo = re.sub(
        r"^```(?:python)?\s*|\s*```$", "", testo, flags=re.MULTILINE
    ).strip()

    # Parsing diretto dell'intero testo
    try:
        risultato = ast.literal_eval(testo)
        if isinstance(risultato, tuple):
            return risultato
    except (ValueError, SyntaxError):
        pass

    # Ricerca di una tupla racchiusa tra parentesi tonde
    match = re.search(r"\(.*?\)", testo, re.DOTALL)
    if match:
        try:
            risultato = ast.literal_eval(match.group())
            if isinstance(risultato, tuple):
                return risultato
        except (ValueError, SyntaxError):
            pass

    raise ValueError(f"Nessuna tupla valida trovata. \n{testo}")


class CheckRegolaOutput(BaseModel):
    ris: bool = False
    conf: float = 0.0
