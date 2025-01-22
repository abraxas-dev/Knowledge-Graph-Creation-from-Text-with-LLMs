1. Lese die Datei `integrator_params.yaml` ein
2. Erstelle eine Instanz der Klasse `Integrator` mit den Parametern aus `integrator_params.yaml` und mache Input data/raw_triples und output matching config data/generated_triples/results.yaml
3. Führe die Methode process() aufm Integrator
4. Jetzt vergleiche die Datei data/ground_truth/expected_results.yaml mit data/generated_triples/results.yaml und berechne dabei folgende Metriken:
    Fur Triples :
    - Precision
    - Recall
    - F1-Score
    Für Entity:
    - Precision
    - Recall
    - F1-Score
    Für Property:
    - Precision
    - Recall
    - F1-Score
    Für Object:
    - Precision
    - Recall
    - F1-Score
5. Speichere die Ergebnisse in der Datei tests/results/results_integration.yaml