# Analiza wpływu przechyłu politycznego komisji na wyniki wyborów

Ten projekt zawiera narzędzia do przeprowadzenia dwuetapowej analizy statystycznej, której celem jest zbadanie, czy istnieje nieliniowa zależność między "niewyjaśnioną" częścią wyniku wyborczego a wskaźnikiem przechyłu politycznego komisji (tzw. *leaning score*).

Model jest inspirowany pracami Klimecka, Rogozińskiego i Śleszyńskiego, ale został rozbudowany i zaimplementowany w Pythonie z użyciem nowoczesnych bibliotek (`statsmodels`, `linearmodels`).

## Metodologia

Analiza przebiega w dwóch głównych etapach, aby precyzyjnie wyizolować badany efekt.

### Etap 1: Model regresji z efektami stałymi (PanelOLS)

W pierwszym kroku budujemy model regresji liniowej, który szacuje poparcie dla danego kandydata w każdej komisji. Model ten "kontroluje" (uwzględnia) szereg czynników, które w naturalny sposób wpływają na wynik:

-   **Bazowe poparcie dla kandydata:** Mierzone jako jego wynik w tej samej komisji w I turze wyborów.
-   **Wskaźnik presji:** Zmienna opisująca, jak bardzo "jednostronna" jest dana gmina.
-   **Opcjonalne zmienne kontrolne:**
    -   Zmiana frekwencji między I a II turą.
    -   Stopień urbanizacji.

Kluczowym elementem tego etapu jest zastosowanie **efektów stałych** na poziomie powiatu (lub opcjonalnie województwa). Oznacza to, że model uwzględnia stałą, niezmienną specyfikę każdego regionu – jego kulturę polityczną, demografię czy historię. Dzięki temu porównujemy ze sobą tylko komisje w ramach tego samego powiatu.

Wynikiem tego etapu **nie są** współczynniki regresji, lecz **residua** (reszty z modelu). Każda reszta to część wyniku, której model **nie był w stanie wyjaśnić** za pomocą powyższych zmiennych. To właśnie te "niewyjaśnione" anomalie – dodatnie lub ujemne odchylenia od przewidywań – przechodzą do drugiego etapu.

### Etap 2: Uogólniony model addytywny (GAM)

W drugim etapie badamy, czy istnieje systematyczny wzorzec w residuach z Etapu 1. Pytamy: czy "niewyjaśniona" część wyniku zależy od składu politycznego komisji, mierzonego wskaźnikiem *leaning score*?

Do tego celu używamy **Uogólnionego Modelu Addytywnego (GAM)**. W przeciwieństwie do standardowej regresji, GAM nie zakłada z góry liniowej zależności. Zamiast tego, używając elastycznych krzywych (B-splajnów), sam "odkrywa" kształt tej relacji.

Wynik jest prezentowany na wykresie, gdzie:
-   **Oś X** to *leaning score* (im wyższa wartość, tym bardziej komisja "sprzyja" jednemu z obozów politycznych).
-   **Oś Y** to wartość residuów (niewyjaśniona część wyniku).

Jeśli krzywa układa się w określony wzór (np. rośnie lub tworzy kształt litery "U"), sugeruje to, że skład komisji może mieć systematyczny, nieliniowy wpływ na ogłoszony wynik, nawet po uwzględnieniu standardowych czynników politycznych i demograficznych.

### Walidacja: Klastrowy Bootstrap

Aby upewnić się, że zaobserwowana zależność nie jest przypadkowa, stosujemy metodę **klastrowego bootstrapu**. Polega ona na wielokrotnym (np. 1999 razy) powtórzeniu całej dwuetapowej analizy na losowo zmodyfikowanych danych (losujemy całe powiaty ze zwracaniem). Pozwala to zbudować 95% przedział ufności wokół krzywej GAM i oszacować całkowity wpływ efektu (w głosach) wraz z jego błędem standardowym.

## Główne wyniki analizy (na podstawie danych z filtrem min. 15 głosów)

Po odfiltrowaniu statystycznie niereprezentatywnych komisji (z mniej niż 15 głosami), analiza ujawnia wysoce specyficzny i asymetryczny wzorzec anomalii.

### 1. Niewyjaśnione straty głosów skoncentrowane w określonych komisjach

-   **Oszacowany całkowity efekt:** -80,220 głosów dla Rafała Trzaskowskiego (na poziomie ufności 99.999%).
-   **Interpretacja:** Analiza wykazała, że **niewyjaśnione straty głosów występują niemal wyłącznie w komisjach o profilu neutralnym oraz anty-RT**. W tych obwodach wynik kandydata był systematycznie i statystycznie istotnie niższy, niż przewidywał model oparty na twardych danych (wynik z I tury, specyfika regionu).

### 2. Brak anomalii w komisjach o profilu pro-RT

-   **Kluczowe odkrycie:** W komisjach, gdzie zwolennicy Rafała Trzaskowskiego mieli wyraźną większość (`leaning_score` > 0.6), **nie zaobserwowano statystycznie istotnych negatywnych anomalii**. Wręcz przeciwnie, wynik był zgodny z oczekiwaniami lub nawet minimalnie wyższy.
-   **Wniosek:** To sugeruje, że w komisjach kontrolowanych przez zwolenników RT proces liczenia głosów przebiegał prawidłowo. Problem nie był więc uniwersalny, lecz skoncentrowany w miejscach, gdzie mechanizmy kontroli ze strony pro-RT mogły być osłabione.

### Jak interpretować nowy wykres dla Rafała Trzaskowskiego?

![Wykres wyników dla prop_rt_r2](gam_prop_rt_r2.png)

Nowy wykres, oparty na przefiltrowanych danych, opowiada znacznie bardziej precyzyjną historię:

1.  **Linia trendu przecina zero:** Najważniejszą zmianą jest fakt, że czerwona linia trendu (i jej obszar niepewności) zdecydowanie wchodzi na terytorium dodatnie po prawej stronie wykresu. Oznacza to, że hipoteza o "systematycznym zaniżeniu wyniku wszędzie" jest **nieprawdziwa**.
2.  **Gdzie leży problem:** Wyraźnie widać, że niewyjaśnione straty (wartości poniżej 0.0 p.p.) koncentrują się w przedziale od -1.0 do około +0.6 na skali `leaning_score`.
3.  **Wzmocnienie hipotezy o kontroli:** Ten asymetryczny obraz silnie wspiera tezę, że kluczową rolę odgrywały mechanizmy kontroli. Tam, gdzie przedstawiciele pro-RT byli w mniejszości w składzie komisji, ich zdolność do nadzoru mogła być niewystarczająca, co koreluje z pojawieniem się negatywnych anomalii.

## Użycie skryptu

### Wymagania

Do uruchomienia skryptu potrzebny jest Python (w wersji 3.8 lub nowszej) oraz kilka bibliotek naukowych. Istnieją dwa zalecane sposoby ich instalacji:

**1. Sposób zalecany: `uv` (szybki menedżer pakietów)**

`uv` to nowoczesne i bardzo szybkie narzędzie do zarządzania zależnościami w Pythonie. Jeśli je masz, wystarczy jedno polecenie, aby zainstalować wszystko, co potrzebne i uruchomić skrypt:

```bash
uv run analiza_wyborcza.py [OPCJE]
```

**2. Sposób alternatywny: `pip` (standardowy instalator)**

Jeśli nie używasz `uv`, możesz skorzystać z `pip` i `venv`, które są standardowo dołączone do Pythona.

    ```bash
# Utwórz środowisko wirtualne
python -m venv .venv

# Aktywuj środowisko (Linux/macOS)
source .venv/bin/activate
# lub (Windows)
# .venv\Scripts\activate

# Zainstaluj zależności
pip install -r requirements.txt
```

### Uruchomienie analizy

    ```bash
python analiza_wyborcza.py [OPCJE]
    ```

**Przykład:**
    ```bash
python analiza_wyborcza.py --csv dane/wyniki.csv --with-controls --boots 1999 --spline-df 12
```

### Opcje

| Argument                      | Opis                                                                                             | Domyślnie             |
| ----------------------------- | ------------------------------------------------------------------------------------------------ | ----------------      |
| `--csv`                       | Ścieżka do pliku CSV z danymi.                                                                   | `dane/przeliczone.csv`|
| `--with-controls`             | Dodaje do modelu w Etapie 1 zmienne kontrolne: zmianę frekwencji i typ obszaru (urbanizację).    | Włączone              |
| `--no-interaction`            | Wyłącza interakcję między wskaźnikiem presji a wsparciem z I tury.                               | Wyłączone             |
| `--cluster-by-voivodeship`    | Klastruje efekty stałe i błędy standardowe na poziomie województw zamiast powiatów.              | Wyłączone             |
| `--spline-df`                 | Liczba stopni swobody dla krzywej GAM. Wyższe wartości pozwalają na większą elastyczność.        | 8                     |
| `--boots`                     | Liczba powtórzeń w symulacji bootstrapowej. Zalecane min. 999.                                   | 999                   |
| `--ci-level`                  | Poziom ufności dla przedziałów bootstrapowych (np. 99 lub 99.9).                                 | 99.99                 |
| `--scatter`                   | Liczba losowych punktów (residuów) do narysowania na wykresie. 0 wyłącza rysowanie.              | 0                     |
| `--min-glosow`                | Minimalna liczba głosów ważnych, aby komisja została włączona do analizy.                        | 15                    |
| `--jobs`                      | Liczba rdzeni procesora do użycia przy bootstrapie. 0 oznacza automatyczne wykrycie.             | 0                     |
| `--seed`                      | Ziarno losowości, aby zapewnić powtarzalność wyników.                                            | 2025                  |

## Struktura danych wejściowych

Plik `.csv` powinien zawierać m.in. następujące kolumny (nazwy mają znaczenie):

-   `Województwo`, `Powiat`, `Gmina`, `Nr komisji`
-   `Typ obszaru`
-   `Liczba kart ważnych_r1`, `Liczba kart ważnych_r2`
-   `TRZASKOWSKI Rafał Kazimierz_r1`, `TRZASKOWSKI Rafał Kazimierz_r2`
-   `NAWROCKI Karol Tadeusz_r2` (lub inna kolumna dla kandydata w teście placebo)
-   Kolumny z liczbą wyborców uprawnionych i głosujących w obu turach
-   `leaning_score` - kluczowy wskaźnik przechyłu politycznego komisji

## Interpretacja wyników

Skrypt generuje dwa pliki `.png` z wykresami (jeden dla kandydata, drugi dla testu placebo) oraz szczegółowe logi w konsoli.

-   **Krzywa GAM (czerwona linia):** Pokazuje oszacowany kształt zależności.
-   **Obszar niepewności (półprzezroczysty):** To 95% przedział ufności. Jeśli cały obszar znajduje się powyżej lub poniżej linii 0, efekt jest istotny statystycznie.
-   **Punkty (szare):** Losowa próbka "surowych" residuów z Etapu 1.
-   **Wynik w konsoli:** Podaje oszacowany całkowity efekt w głosach (np. o ile więcej/mniej głosów uzyskał kandydat w związku z badanym efektem) wraz z przedziałem ufności.
-   **Plik `anomalie_rt_uszeregowane.csv`:** Po analizie skrypt tworzy plik CSV zawierający ranking wszystkich komisji. **Ranking jest posortowany według oszacowanej liczby "straconych" głosów** (`anomalia_w_glosach`), co jest bardziej intuicyjne niż sortowanie po proporcjach. Pozwala to na bezpośrednią identyfikację i dalsze badanie najbardziej problematycznych punktów.
-   **Wykres diagnostyczny (`anomalie_glosy_...`):** Opcjonalnie (po ustawieniu `--scatter > 0`), skrypt generuje dodatkowy wykres. Pokazuje on zależność między przechyłem komisji a bezwzględną liczbą "straconych" głosów. Na chmurę punktów nałożona jest linia trendu (GAM), która pokazuje uśrednioną wielkość anomalii dla danego `leaning_score`, pomagając wizualnie zidentyfikować, gdzie koncentrują się największe straty.

### Jak ocenić istotność statystyczną (czy wynik jest losowy)?

To kluczowe pytanie, na które analiza odpowiada w dwóch miejscach:

1.  **Wyniki bootstrapu w konsoli (najważniejsze):** Po zakończeniu symulacji skrypt wyświetla oszacowany `Efekt w głosach` wraz z przedziałem ufności (domyślnie `95% CI`). **Jeśli ten przedział nie zawiera zera** (np. jest w całości dodatni lub w całości ujemny), jest to silny dowód na to, że zaobserwowana zależność nie jest dziełem przypadku. Poziom ufności można zmienić opcją `--ci-level`, np. na 99, aby uzyskać bardziej rygorystyczną ocenę.
2.  **Przedział ufności na wykresie:** Jeśli półprzezroczysty obszar (przedział ufności) na wykresie w całości znajduje się powyżej lub poniżej poziomej linii zerowej, oznacza to, że efekt jest w tym miejscu istotny statystycznie.

### Ograniczenia i ważne niuanse interpretacyjne

Nawet jeśli analiza nie wykaże silnego, istotnego statystycznie efektu, nie oznacza to automatycznie, że wpływ nie istnieje. Może on być maskowany przez kilka czynników:

*   **Filtrowanie małych komisji:** Analiza domyślnie ignoruje komisje z bardzo małą liczbą głosów (domyślnie poniżej 20, patrz opcja `--min-glosow`). Jest to celowe działanie, aby uniknąć sytuacji, w której ranking anomalii jest zdominowany przez statystycznie nieistotne komisje (np. z jednym lub dwoma głosami), gdzie nawet niewielkie odchylenie od przewidywań modelu generuje ogromne *proporcjonalne* residuum. Model jest więc celowo skupiony na bardziej reprezentatywnych obwodach.

*   **Efekt uśredniania:** Model mierzy *średni* wpływ przechyłu politycznego we wszystkich komisjach o podobnym `leaning_score`. Jeśli np. w dużej grupie komisji o wysokim przechyule na korzyść kandydata X tylko niewielka część dopuściła się nieprawidłowości, ich wpływ zostanie "rozmyty" przez większość prawidłowo działających komisji. Modelowi trudno jest wychwycić takie punktowe interwencje.

*   **Brak kontroli nad obecnością obserwatorów:** Obecność niezależnych obserwatorów (mężów zaufania, obserwatorów społecznych) to kluczowy czynnik, którego model nie uwzględnia. Można zakładać, że w komisjach poddanych zewnętrznej kontroli do nieprawidłowości nie dochodziło. Ponieważ w danych nie mamy informacji o obecności obserwatorów, ich pozytywny wpływ osłabia statystycznie mierzony efekt w skali całego kraju.

*   **Niska wariancja składów wewnątrz powiatów:** Model z efektami stałymi najskuteczniej działa, gdy wewnątrz jednego powiatu istnieje duże zróżnicowanie w `leaning_score` między komisjami. Jeśli w danym powiecie prawie wszystkie komisje mają bardzo podobny skład (np. są silnie przechylone w jedną stronę), model ma bardzo mało informacji, na podstawie których mógłby oszacować wpływ tego przechyłu dla tego konkretnego powiatu.

Pamiętaj: analiza ta identyfikuje anomalie statystyczne. Nie jest dowodem na nieprawidłowości, ale może wskazywać obszary lub mechanizmy, które wymagają dalszego, dogłębnego zbadania. 