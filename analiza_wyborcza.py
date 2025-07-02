#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analiza_wyborcza.py: Zaawansowana, dwuetapowa analiza danych wyborczych.

Skrypt implementuje dwuetapową procedurę estymacji w celu zbadania
potencjalnych nieliniowych zależności między "niewyjaśnioną" częścią wyniku
wyborczego a wskaźnikiem politycznego przechyłu komisji ("leaning score").

Metodologia:
1.  **Etap 1: PanelOLS z efektami stałymi**
    - Modeluje poparcie dla kandydata, kontrolując jego bazowe poparcie (z I tury),
      wskaźnik presji oraz (opcjonalnie) interakcję i dodatkowe zmienne.
    - Efekty stałe na poziomie powiatu/województwa "oczyszczają" dane
      ze stałej, niewidocznej specyfiki regionów.
    - Kluczowym wynikiem są residua (reszty z modelu), które stanowią
      "niewyjaśnioną" część wyniku i przechodzą do Etapu 2.

2.  **Etap 2: Uogólniony Model Addytywny (GAM)**
    - Bada, czy residua z Etapu 1 układają się w nieliniowy wzorzec
      w zależności od "leaning score".
    - Wykorzystuje elastyczne krzywe (B-splajny) do modelowania tej zależności.

Wiarygodność wyników jest oceniana za pomocą **klastrowego bootstrapu**,
który jest złotym standardem w tego typu analizach. Generowane są tysiące
alternatywnych próbek danych (poprzez losowanie całych klastrów), co pozwala
na oszacowanie stabilności modelu i skonstruowanie 95% przedziałów ufności.

Wyniki należy interpretować z ostrożnością. Skrypt identyfikuje anomalie
statystyczne, a nie udowadnia nieprawidłowości.

Użycie:
```bash
uv run analiza_wyborcza.py
```
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import statsmodels.api as sm
from statsmodels.gam.api import BSplines, GLMGam
from linearmodels.panel import PanelOLS
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties

###############################################################################
# Data helpers
###############################################################################

def read_csv_auto(path: Path) -> pd.DataFrame:
    for sep in (";", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 5:
                return df
        except Exception:
            pass
    raise SystemExit(f"Cannot read '{path}' with ; or , delimiter.")


def prepare_frame(df_raw: pd.DataFrame, args) -> pd.DataFrame:
    """Zwraca ramkę danych gotową do analizy."""
    df = df_raw.copy()
    if args.cluster_by_voivodeship:
        df["jednostka_analizy"] = df["Województwo"]
        df.loc[df["Gmina"] == "zagranica", "jednostka_analizy"] = "zagranica"
    else:
        df["jednostka_analizy"] = df["Powiat"]
        df.loc[df["Gmina"] == "zagranica", "jednostka_analizy"] = df["Kraj"]

    df["lacznie_glosow_r2"] = df["Liczba kart ważnych_r2"].fillna(0).astype(int)

    # Filtrowanie komisji o bardzo małej liczbie głosów
    print(f"Liczba komisji przed filtrowaniem: {len(df):,}")
    df = df[df["lacznie_glosow_r2"] >= args.min_votes].copy()
    print(f"Liczba komisji po odfiltrowaniu tych z < {args.min_votes} głosami: {len(df):,}")

    df["prop_rt_r2"] = (
        df["TRZASKOWSKI Rafał Kazimierz_r2"].fillna(0).astype(int)
        / df["lacznie_glosow_r2"]
    )
    df["prop_kn_r2"] = (
        df["NAWROCKI Karol Tadeusz_r2"].fillna(0).astype(int) / df["lacznie_glosow_r2"]
    )

    df["wsparcie_rt_r1"] = (
        df["TRZASKOWSKI Rafał Kazimierz_r1"].clip(lower=1)
        / df["Liczba kart ważnych_r1"].clip(lower=1)
    )
    df["wskaznik_presji"] = (1 - df["leaning_score"]) / 2
    df["presja_x_wsparcie"] = df["wskaznik_presji"] * df["wsparcie_rt_r1"]

    # optional controls
    r1_out, r1_elig = (
        "Liczba wyborców, którym wydano karty do głosowania w lokalu wyborczym oraz w głosowaniu korespondencyjnym (łącznie)_r1",
        "Liczba wyborców uprawnionych do głosowania (umieszczonych w spisie, z uwzględnieniem dodatkowych formularzy) w chwili zakończenia głosowania_r1",
    )
    r2_out, r2_elig = (
        "Liczba wyborców, którym wydano karty do głosowania w lokalu wyborczym oraz w głosowaniu korespondencyjnym (łącznie)_r2",
        "Liczba wyborców uprawnionych do głosowania (umieszczonych w spisie, z uwzględnieniem dodatkowych formularzy) w chwili zakończenia głosowania_r2",
    )
    df["frekwencja_r1"] = df[r1_out] / df[r1_elig]
    df["frekwencja_r2"] = df[r2_out] / df[r2_elig]
    df["zmiana_frekwencji"] = df["frekwencja_r2"] - df["frekwencja_r1"]
    df["urbanizacja"] = pd.Categorical(df["Typ obszaru"]).codes

    req_cols = [
        "prop_rt_r2",
        "prop_kn_r2",
        "lacznie_glosow_r2",
        "wsparcie_rt_r1",
        "wskaznik_presji",
        "presja_x_wsparcie",
        "zmiana_frekwencji",
        "urbanizacja",
        "jednostka_analizy",
        "Nr komisji",
        "leaning_score",
    ]
    return (
        df.dropna(subset=req_cols)
        .assign(jednostka_analizy=lambda d: d["jednostka_analizy"].astype(str))
    )

###############################################################################
# Fixed‑effects stage
###############################################################################

def fit_fe_and_residuals(df: pd.DataFrame, y_col: str, with_ctrl: bool, with_interaction: bool) -> np.ndarray:
    """Uruchamia model OLS z efektami stałymi i zwraca residua."""
    idx = df.set_index(["jednostka_analizy", "Nr komisji"], drop=False)
    y = idx[y_col]

    zmienne_x = ["wsparcie_rt_r1", "wskaznik_presji"]
    if with_interaction:
        zmienne_x.append("presja_x_wsparcie")

    if with_ctrl:
        zmienne_x += ["zmiana_frekwencji", "urbanizacja"]

    X = sm.add_constant(idx[zmienne_x])
    model_fe = PanelOLS(y, X, entity_effects=True, weights=idx["lacznie_glosow_r2"])
    residua = model_fe.fit(cov_type="clustered", cluster_entity=True).resids
    return residua.to_numpy()

###############################################################################
# GAM helpers
###############################################################################

def fit_gam(df: pd.DataFrame, kol_resid: str, df_spline: int) -> GLMGam:
    """Dopasowuje model GAM do residuów."""
    bs = BSplines(df[["leaning_score"]], df=[df_spline], degree=3, include_intercept=True)
    return GLMGam.from_formula(
        f"{kol_resid} ~ leaning_score",
        data=df,
        smoother=bs,
        weights=df["lacznie_glosow_r2"],
    ).fit()

###############################################################################
# Bootstrap worker
###############################################################################

def bootstrap_rep(df_orig: pd.DataFrame, kandydat_col: str, kol_resid: str, args, grid, seed):
    """Pojedyncza iteracja bootstrapu klastrowego."""
    rng = np.random.default_rng(seed)

    # Losowanie z powtórzeniami całych klastrów (jednostek analizy)
    klastry = df_orig["jednostka_analizy"].unique()
    losowe_klastry = rng.choice(klastry, size=len(klastry), replace=True)

    # Tworzenie nowej ramki danych na podstawie wylosowanych klastrów
    rep_blocks = []
    for j, cid in enumerate(losowe_klastry):
        block = df_orig[df_orig["jednostka_analizy"] == cid].copy()
        # Unikalne ID dla powtórzonych klastrów
        block["jednostka_analizy"] = f"{cid}_{j}"
        rep_blocks.append(block)
    df_rep = pd.concat(rep_blocks, ignore_index=True)

    # Krok 1: Oblicz residua dla nowej, zresamplowanej ramki danych
    df_rep[kol_resid] = fit_fe_and_residuals(df_rep, kandydat_col, args.with_controls, not args.no_interaction)
    df_rep.dropna(subset=["leaning_score", kol_resid], inplace=True)
    if df_rep.empty:
        return None, None, None, None

    # Krok 2: Dopasuj model GAM do tych residuów
    bs_rep = BSplines(df_rep[["leaning_score"]], df=[args.spline_df], degree=3, include_intercept=True)
    gam_rep = GLMGam.from_formula(
        f"{kol_resid} ~ leaning_score",
        data=df_rep,
        smoother=bs_rep,
        weights=df_rep["lacznie_glosow_r2"],
    ).fit()

    # Przewiduj wartości na siatce, aby zbudować przedział ufności dla krzywej
    rep_min, rep_max = df_rep["leaning_score"].min(), df_rep["leaning_score"].max()
    grid_clip = grid.copy()
    grid_clip["leaning_score"] = grid_clip["leaning_score"].clip(rep_min, rep_max)
    pred_curve = gam_rep.get_prediction(grid_clip, exog_smooth=grid_clip).predicted_mean

    # Oszacuj całkowity "przesunięty" wolumen głosów w tej replikacji bootstrapowej
    xs_full = df_orig[["leaning_score"]].copy()
    xs_full["leaning_score"] = xs_full["leaning_score"].clip(rep_min, rep_max)
    full_pred = gam_rep.get_prediction(xs_full, exog_smooth=xs_full).predicted_mean
    vote_shift = (full_pred * df_orig["lacznie_glosow_r2"]).sum()

    # Oszacuj stratę głosów względem benchmarku leaning_score=1.0
    benchmark_df = pd.DataFrame({"leaning_score": [1.0]})
    benchmark_df["leaning_score"] = benchmark_df["leaning_score"].clip(rep_min, rep_max)
    benchmark_residual = gam_rep.get_prediction(benchmark_df, exog_smooth=benchmark_df).predicted_mean[0]

    predicted_residuals = gam_rep.get_prediction(xs_full, exog_smooth=xs_full).predicted_mean
    deficit = benchmark_residual - predicted_residuals
    
    vote_loss_df = pd.DataFrame({
        'deficit': deficit,
        'votes': df_orig["lacznie_glosow_r2"]
    })
    vote_loss_df['vote_loss'] = vote_loss_df['deficit'] * vote_loss_df['votes']
    total_vote_loss = vote_loss_df[vote_loss_df['vote_loss'] > 0]['vote_loss'].sum()
    
    deficit_curve = benchmark_residual - pred_curve
    
    return pred_curve, vote_shift, total_vote_loss, deficit_curve

###############################################################################
# Plotting helper
###############################################################################

def generate_and_save_plots(curves, deficit_curves, gam_full, grid, kandydat_col, args, iteration):
    """Generuje i zapisuje wykresy wyników bootstrapu."""
    
    # Obliczanie granic przedziału ufności
    alpha = (100 - args.ci_level) / 2
    ci_low_perc = alpha
    ci_high_perc = 100 - alpha

    ci_low_curve = np.percentile(curves, ci_low_perc, axis=0)
    ci_high_curve = np.percentile(curves, ci_high_perc, axis=0)

    # Obliczenie głównej predykcji
    pred_mean = gam_full.get_prediction(grid, exog_smooth=grid).predicted_mean

    # --- Wykres główny ---
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(grid["leaning_score"], pred_mean, color="#d62728", lw=4, label="Uśredniony trend")
    ax.fill_between(grid["leaning_score"], ci_low_curve, ci_high_curve, color="red", alpha=0.15, label=f"Obszar niepewności ({args.ci_level}%)")
    ax.set_title("Asymetryczny wzorzec niewyjaśnionych anomalii", fontsize=22, pad=20, weight='bold')
    ax.set_xlabel("Skład polityczny komisji (od anty-RT do pro-RT)", fontsize=14)
    ax.set_ylabel("Odchylenie od oczekiwanego wyniku RT (w p.p.)", fontsize=14)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.1f} p.p.'))
    ax.text(0.05, 0.1, 'W komisjach o profilu anty-RT\nlub nieznacznie pro-RT\nwynik RT był systematycznie zaniżony.',
            transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    ax.text(0.7, 0.8, 'W komisjach o silnym profilu pro-RT\nwynik RT był powyżej oczekiwań\n(opartych na ogłoszonych wynikach).',
            transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7))
    ax.axhline(0, color="black", lw=1.5, linestyle='--')
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    
    out_name = f"gam_{kandydat_col}_iter_{iteration}.png"
    plt.savefig(out_name, dpi=300)
    print(f"Zapisano wykres: {out_name}")
    plt.close(fig)

    # --- Wykres straty względem L=1 (tylko dla głównego kandydata) ---
    if kandydat_col == "prop_rt_r2":
        ci_low_deficit = np.percentile(deficit_curves, ci_low_perc, axis=0)
        ci_high_deficit = np.percentile(deficit_curves, ci_high_perc, axis=0)
        
        benchmark_df_full = pd.DataFrame({"leaning_score": [1.0]})
        benchmark_resid_full = gam_full.get_prediction(benchmark_df_full, exog_smooth=benchmark_df_full).predicted_mean[0]
        deficit_curve_full = benchmark_resid_full - pred_mean

        fig_loss, ax_loss = plt.subplots(figsize=(18, 10))
        ax_loss.plot(grid["leaning_score"], deficit_curve_full * 100, color="#2ca02c", lw=4, label="Uśredniona strata wyniku")
        ax_loss.fill_between(grid["leaning_score"], ci_low_deficit * 100, ci_high_deficit * 100, color="#2ca02c", alpha=0.15, label=f"Obszar niepewności ({args.ci_level}%)")
        ax_loss.set_title("Szacowana strata wyniku RT względem komisji w pełni pro-RT (L=1)", fontsize=22, pad=20, weight='bold')
        ax_loss.set_xlabel("Skład polityczny komisji (od anty-RT do pro-RT)", fontsize=14)
        ax_loss.set_ylabel("Strata wyniku (w p.p.)", fontsize=14)
        ax_loss.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.1f} p.p.'))
        ax_loss.axhline(0, color="black", lw=1.5, linestyle='--')
        ax_loss.legend(loc="best", fontsize=12)
        ax_loss.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        ax_loss.spines[['top', 'right']].set_visible(False)
        fig_loss.tight_layout()
        
        out_name_loss = f"gam_strata_{kandydat_col}_iter_{iteration}.png"
        plt.savefig(out_name_loss, dpi=300)
        print(f"Zapisano wykres straty: {out_name_loss}")
        plt.close(fig_loss)

###############################################################################
# Per-candidate routine
###############################################################################

def analyse_candidate(df: pd.DataFrame, kandydat_col: str, args):
    """Uruchamia pełną, dwuetapową analizę dla wybranego kandydata."""
    print(f"\n{'='*80}\nANALIZA DLA: {kandydat_col.upper()}\n{'='*80}")
    kol_resid = f"resid_{kandydat_col}"
    df[kol_resid] = fit_fe_and_residuals(df, kandydat_col, args.with_controls, not args.no_interaction)

    # Zachowaj pełny zbiór z residuami do rankingu i wykresu diagnostycznego
    df_with_residuals = df.copy()
    
    # Usuń brakujące dane na potrzeby głównej analizy GAM i bootstrapu
    df.dropna(subset=["leaning_score", kol_resid], inplace=True)
    if df.empty:
        print("Brak danych po odfiltrowaniu, przerywam analizę dla tego kandydata.")
        return

    # Zdefiniuj siatkę na podstawie przefiltrowanych danych
    grid = pd.DataFrame({"leaning_score": np.linspace(df["leaning_score"].min(), df["leaning_score"].max(), 250)})

    # Zapis rankingu anomalii i dodatkowy wykres (tylko dla głównego kandydata)
    if kandydat_col == "prop_rt_r2":
        print("\n--- ZAPIS RANKINGU ANOMALII ---")
        df_to_save = df_with_residuals  # Użyj pełnego zbioru

        # Oblicz anomalię w głosach bezwzględnych - nowa metryka do rankingu
        kol_anomalia_glosy = "anomalia_w_glosach"
        df_to_save[kol_anomalia_glosy] = df_to_save[kol_resid] * df_to_save["lacznie_glosow_r2"]

        # Sortuj według nowej, bardziej intuicyjnej metryki
        df_to_save.sort_values(by=kol_anomalia_glosy, ascending=True, inplace=True)
        
        # Przeniesienie kluczowych kolumn na początek dla czytelności
        first_cols = [kol_anomalia_glosy, kol_resid, "leaning_score", "Województwo", "Powiat", "Gmina", "Nr komisji"]
        other_cols = [c for c in df_to_save.columns if c not in first_cols]
        df_to_save = df_to_save[first_cols + other_cols]
        
        out_csv_name = "anomalie_rt_uszeregowane.csv"
        try:
            df_to_save.to_csv(out_csv_name, index=False, sep=';', decimal=',')
            print(f"Zapisano ranking komisji do pliku: {out_csv_name}")
        except Exception as e:
            print(f"Błąd przy zapisie pliku CSV: {e}")

    gam_full = fit_gam(df, kol_resid, args.spline_df)
    print("\n--- GAM (pełny zbiór) ---")
    print(gam_full.summary())

    # Bootstrap
    print(f"\nBootstrap klastrowy ({args.boots} replik)…")
    workers = args.jobs if args.jobs > 0 else max(1, os.cpu_count() - 1)
    
    all_curves, all_shifts, all_losses, all_deficit_curves = [], [], [], []
    
    save_interval = args.save_interval
    if save_interval <= 0:
        save_interval = args.boots
    
    seeds = range(args.boots)

    for i in tqdm(range(0, args.boots, save_interval), desc="Bootstrap batches"):
        batch_seeds = seeds[i : i + save_interval]
        if not batch_seeds:
            break
            
        rep_out = Parallel(n_jobs=workers, verbose=0)(
            delayed(bootstrap_rep)(df, kandydat_col, kol_resid, args, grid, seed)
            for seed in batch_seeds
        )
        
        # Unpack and filter results from the current batch
        batch_curves, batch_shifts, batch_losses, batch_deficit_curves = zip(*[r for r in rep_out if r[0] is not None])
        if not batch_curves:
            continue
        
        all_curves.extend(batch_curves)
        all_shifts.extend(batch_shifts)
        all_losses.extend(batch_losses)
        all_deficit_curves.extend(batch_deficit_curves)
        
        # Convert to numpy arrays for processing
        np_curves = np.vstack(all_curves)
        np_deficit_curves = np.vstack(all_deficit_curves)
        
        num_reps_done = len(all_shifts)
        
        if args.save_interval > 0:
            print(f"\n--- Zapisywanie wykresów po {num_reps_done} iteracjach ---")
            generate_and_save_plots(
                np_curves, np_deficit_curves, gam_full, grid, kandydat_col, args, iteration=num_reps_done
            )

    # Końcowe podsumowanie
    np_shifts = np.array(all_shifts)
    np_losses = np.array(all_losses)
    
    # Obliczanie granic przedziału ufności na podstawie argumentu --ci-level
    alpha = (100 - args.ci_level) / 2
    ci_low_perc = alpha
    ci_high_perc = 100 - alpha

    vote_point = np_shifts.mean()
    ci_low_shift, ci_high_shift = np.percentile(np_shifts, [ci_low_perc, ci_high_perc])
    mc_se = np_shifts.std(ddof=1) / np.sqrt(len(np_shifts))

    loss_point = np_losses.mean()
    ci_low_loss, ci_high_loss = np.percentile(np_losses, [ci_low_perc, ci_high_perc])
    mc_se_loss = np_losses.std(ddof=1) / np.sqrt(len(np_losses))

    print("\n--- WYNIKI BOOTSTRAPU (KOŃCOWE) ---")
    print(f"Liczba udanych replik: {len(np_shifts):,}")
    print(f"Efekt netto (przesunięcie)       : {vote_point:,.0f}   ({args.ci_level}% CI: {ci_low_shift:,.0f} do {ci_high_shift:,.0f})")
    print(f"Błąd standardowy MC (przesunięcie): ±{mc_se:,.0f} głosów")
    print(f"Szacowana strata głosów (vs L=1) : {loss_point:,.0f}   ({args.ci_level}% CI: {ci_low_loss:,.0f} do {ci_high_loss:,.0f})")
    print(f"Błąd standardowy MC (strata)     : ±{mc_se_loss:,.0f} głosów")

    # Zapis ostatecznych wykresów
    print("\n--- Zapisywanie ostatecznych wykresów ---")
    final_np_curves = np.vstack(all_curves)
    final_np_deficit_curves = np.vstack(all_deficit_curves)
    generate_and_save_plots(
        final_np_curves, final_np_deficit_curves, gam_full, grid, kandydat_col, args, iteration=f"{len(np_shifts)}_final"
    )

###############################################################################
# CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Dwustopniowa analiza zależności między wynikiem wyborczym a przechyłem politycznym komisji.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default="dane/przeliczone.csv", help="Ścieżka do pliku CSV z danymi")
    parser.add_argument("--with-controls", default=True, action="store_true", help="Dodaj do modelu kontrolę zmiany frekwencji i urbanizacji")
    parser.add_argument("--no-interaction", action="store_true", help="Wyłącz interakcję między presją a wsparciem z I tury")
    parser.add_argument("--cluster-by-voivodeship", action="store_true", help="Klastruj na poziomie województw (domyślnie: powiaty)")
    parser.add_argument("--spline-df", type=int, default=8, help="Liczba stopni swobody dla krzywych sklejanych (splajnów)")
    parser.add_argument("--boots", type=int, default=999, help="Liczba replikacji bootstrapu klastrowego")
    parser.add_argument("--ci-level", type=float, default=99.99, help="Poziom ufności dla przedziałów (np. 95, 99)")
    parser.add_argument("--save-interval", type=int, default=100, help="Co ile iteracji bootstrapu zapisywać częściowe wykresy (0=tylko na końcu)")
    parser.add_argument("--scatter", type=int, default=0, help="Liczba punktów do narysowania na wykresie rozrzutu (0=wyłącz)")
    parser.add_argument("--min-votes", type=int, default=15, help="Minimalna liczba głosów ważnych, aby komisja została włączona do analizy")
    parser.add_argument("--jobs", type=int, default=0, help="Liczba rdzeni CPU do użycia (0=automatycznie)")
    parser.add_argument("--seed", type=int, default=2025, help="Ziarno losowości dla powtarzalności wyników")
    args = parser.parse_args()

    # Walidacja poziomu ufności
    if not 0 < args.ci_level < 100:
        raise SystemExit("Błąd: --ci-level musi być wartością z przedziału (0, 100).")

    np.random.seed(args.seed)
    print(f"Seed set to {args.seed}")

    df_raw = read_csv_auto(args.csv)
    df = prepare_frame(df_raw, args)

    analyse_candidate(df.copy(), "prop_rt_r2", args)

    print("\n" + "*" * 80 + "\nTEST SYMETRII (analiza dla kontrkandydata)\n" + "*" * 80)
    analyse_candidate(df.copy(), "prop_kn_r2", args)

    print("\n✓ Analiza zakończona.")

if __name__ == "__main__":
    main()
