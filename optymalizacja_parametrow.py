#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optymalizacja_parametrow.py: Skrypt do analizy wrażliwości i optymalizacji parametrów.

Ten skrypt wykorzystuje bibliotekę Optuna do systematycznego przeszukiwania
przestrzeni parametrów (`spline-df`, `min-votes`) w celu znalezienia kombinacji,
która maksymalizuje (w wartości bezwzględnej) oszacowany "efekt w głosach".

UWAGA METODOLOGICZNA:
To jest narzędzie do EKSPLORACJI, a nie do generowania ostatecznych wyników.
Celem jest zbadanie wrażliwości modelu na parametry wejściowe i wskazanie
"najbardziej obiecującej" kombinacji do dalszej, rygorystycznej analizy.
Wynik znaleziony przez ten skrypt należy traktować jako hipotezę, którą
trzeba następnie zweryfikować, uruchamiając `analiza_wyborcza.py`
z najlepszymi parametrami i dużą liczbą bootstrapów (np. --boots 1999).
"""
import subprocess
import re
import argparse
import optuna
from tqdm import tqdm

# Globalna tqdm progress bar
pbar = None

def parse_effect_from_output(output: str) -> float:
    """Przetwarza wyjście z konsoli i wyciąga 'Efekt w głosach'."""
    match = re.search(r"Efekt w głosach\s+:\s+([-\d,\.]+)", output)
    if match:
        try:
            # Usuń przecinki i konwertuj na float
            effect_str = match.group(1).replace(",", "")
            return float(effect_str)
        except (ValueError, IndexError):
            return float('inf')  # Zwróć nieskończoność w przypadku błędu parsowania
    return float('inf') # Zwróć nieskończoność, jeśli nie znaleziono dopasowania

def objective(trial: optuna.Trial) -> float:
    """
    Definiuje pojedynczą próbę optymalizacji.
    Optuna będzie próbowała zminimalizować wartość zwracaną przez tę funkcję.
    """
    global pbar
    # 1. Sugerowanie parametrów do przetestowania przez Optunę
    spline_df = trial.suggest_int('spline_df', 5, 25)
    min_votes = trial.suggest_int('min_votes', 10, 100)
    
    # 2. Budowanie i uruchamianie polecenia
    command = [
        "python", "analiza_wyborcza.py",
        "--spline-df", str(spline_df),
        "--min-votes", str(min_votes),
        "--boots", "10"  # Używamy małej liczby dla szybkości
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        effect = parse_effect_from_output(output)
    except subprocess.CalledProcessError as e:
        print(f"Błąd podczas uruchamiania analizy: {e.stderr}")
        effect = float('inf') # Duża wartość, aby Optuna unikała tej ścieżki
    
    pbar.update(1)
    return effect


def main():
    global pbar
    parser = argparse.ArgumentParser(description="Optymalizator parametrów dla skryptu analizy wyborczej.")
    parser.add_argument("--trials", type=int, default=100, help="Liczba prób optymalizacji do przeprowadzenia.")
    args = parser.parse_args()

    print("Rozpoczynam optymalizację parametrów...")
    print(f"Liczba prób: {args.trials}")
    print("Celem jest znalezienie parametrów minimalizujących 'Efekt w głosach'.\n")

    # Inicjalizacja progress bar
    pbar = tqdm(total=args.trials)

    # 3. Tworzenie i uruchamianie studium Optuny
    # Kierunek 'minimize', ponieważ szukamy najbardziej negatywnej wartości
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    pbar.close()

    # 4. Prezentacja wyników
    print("\n" + "="*80)
    print("OPTYMALIZACJA ZAKOŃCZONA")
    print("="*80)
    print(f"Liczba ukończonych prób: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\n--- NAJLEPSZE ZNALEZIONE PARAMETRY ---")
    print(f"  Wartość celu (Efekt w głosach): {best_trial.value:,.0f}")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    print("\n" + "-"*80)
    print("REKOMENDACJA:")
    print("Uruchom teraz główną analizę z powyższymi parametrami i dużą liczbą")
    print("bootstrapów, aby uzyskać wiarygodne wyniki, np.:")
    best_params_str = " ".join([f"--{key} {value}" for key, value in best_trial.params.items()])
    print(f"\npython analiza_wyborcza.py {best_params_str} --boots 1999\n")


if __name__ == "__main__":
    main() 