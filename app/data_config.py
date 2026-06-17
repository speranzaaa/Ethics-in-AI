SYSTEM_OVERVIEW = "Il sistema combina due layer che guardano segnali diversi. L'LLM valuta il contenuto clinico della visita, il KDE valuta il pattern temporale degli accessi del paziente. I due sbagliano su casi diversi, quindi sommando i loro score gli errori di uno vengono coperti dall'altro. Nei grafici si vede che nessuno dei due layer preso da solo separa bene le tre classi, mentre la somma le distingue in modo netto e coerente con le soglie di rischio." 

# Grafici mostrati nella vista "Caso totale".

CHARTS = [
    {
        "image": "soloKDE.png",
        "titolo": "KDE score",
        "descrizione": "I casi non-NAP restano tutti a zero, quindi il layer non genera falsi allarmi sui casi normali. Quasi tutti i NAP si concentrano intorno a 0.90. Fanno eccezione il caso 12, più basso a circa 0.57, e il caso 13, che resta vicino a zero: da solo l'LLM lo manca. I sospetti sono divisi, con 5, 6 e 7 tra 0.37 e 0.61 e gli altri due quasi a zero.",  
    },
    {
        "image": "soloLLM.png",
        "titolo": "LLM score",
        "descrizione": "Score del solo layer statistico, calcolato sui pattern temporali degli accessi. Anche qui i non-NAP stanno a zero. Sui NAP il comportamento è quasi opposto a quello dell'LLM: prende bene i casi 12 e 13 (circa 0.75 e 0.80) e tiene il caso 14 intorno a 0.49, ma manca i casi 10 e 11, che restano vicino a zero. Tra i sospetti salgono solo 8 e 9 a circa 0.35, gli altri restano bassi.",  
    },
    {
        "image": "LLM+KDE.png",
        "titolo": "Score totale (KDE + LLM)",
        "descrizione": "Score finale del sistema, somma dei due layer, ed è il valore su cui agiscono le soglie. Le tre categorie si separano senza sovrapposizioni: non-NAP a zero, sospetti tra 0.35 e 0.63, NAP tra 0.79 e 1.40. La somma recupera gli errori dei singoli layer. Il caso 13, quasi a zero nel solo LLM, risale grazie al KDE. I casi 10 e 11, persi dal KDE, risalgono grazie all'LLM.",  
    },
]
