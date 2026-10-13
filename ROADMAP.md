# Quality × Value × Momentum Lab — roadmap

## Finální cíl

Bezpečná, reprodukovatelná veřejná výzkumná aplikace pro screening akcií a testování strategie s nízkou obrátkou portfolia. Aplikace musí umět prokázat, odkud data pocházejí, kdy byla dostupná, jak se počítají signály a jak se mění výsledek po nákladech. Dokud nejsou splněny validační podmínky, nesmí být backtest prezentován jako důkaz budoucí výkonnosti.

## Fáze 0 — Oprava výpočtů a základní QA

- [x] Zkontrolovat logiku současného screeneru a backtestu.
- [x] Srovnat signál backtestu s výnosem následujícího měsíce.
- [x] Použít hrubý objem obchodů pro odhad nákladů.
- [x] Počítat průraz z předchozích denních maxim a ATR(20) z true range.
- [x] Nahradit absolutní FCF relativní marží a penalizovat neúplné fundamentální skóre tím, že se chybějící vstupy ignorují a vykazuje se pokrytí.
- [x] Přidat přehled dostupnosti cenových řad a automatické regresní testy.
- [ ] Zkontrolovat výsledek CI a nasazení na Streamlit Community Cloud.

## Fáze 1 — Ověřená data

- [ ] Přidat automatickou kontrolu stavu zdroje, posledního obchodního dne, prázdných řad, duplicit a neobvykle velkých cenových skoků.
- [ ] U fundamentálních dat zobrazovat zdroj, období výkazu, datum zveřejnění, datum stažení a úspěšnost načtení.
- [ ] Přidat kontrolní vzorek vybraných tickerů porovnaný s nezávislým zdrojem.
- [ ] Pro americké akcie přidat oficiální SEC EDGAR Company Facts/filing data; respektovat datum zveřejnění a opravy výkazů.
- [ ] Pro jiné trhy vybrat zdroj s jasnou licencí, pokrytím a limity. Nezáviset na nezdokumentovaném API jako na jediné produkční databázi.

**Brána fáze:** cena a fundament každé firmy mají identifikovatelný zdroj, datum a stav úplnosti.

## Fáze 2 — Bodově správné fundamenty a signály

- [ ] Zavést point-in-time fundamenty; historická rozhodnutí smějí používat jen informace zveřejněné do daného data.
- [ ] Zlepšit srovnatelnost sektorů a metrik: ROIC/ROE, marže, zadlužení, cash-flow marže a valuation násobky.
- [ ] Oddělit ziskové a ztrátové firmy; záporné násobky neinterpretovat jako levné ocenění.
- [ ] Přidat revize odhadů zisků jen pokud historie a čas zveřejnění jsou doložitelné.
- [ ] Sentiment přidat až po vyřešení historických timestampů, licencí a reprodukovatelnosti.
- [ ] Každý signál popsat vzorcem, jednotkou, obdobím a zacházením s chybějícími daty.

**Brána fáze:** stejné vstupy a stejná konfigurace musí dát přesně stejné skóre; neúplná data nesmí být zaměňována za skutečné neutrální hodnoty.

## Fáze 3 — Backtest bez zásadního úniku informací

- [ ] Použít historický investiční univerzum včetně delistovaných titulů a změn indexových složení.
- [ ] Definovat přesný okamžik signálu, okamžik obchodu a cenu realizace bez použití budoucího close.
- [ ] Zahrnout dividendy, splits, měny, spread, skluz, komise a realistické náklady obratu portfolia.
- [ ] Přidat benchmarky (např. široký tržní ETF odpovídající regionu) a buy-and-hold srovnání.
- [ ] Ověřit hotovost, chybějící ceny, IPO, delistingy, corporate actions a rebalancování.
- [ ] Oddělit vývojové období, walk-forward validaci a předem nedotčený holdout.

**Brána fáze:** testy bez look-ahead bias, srovnání s benchmarkem a výsledky po nákladech jsou reprodukovatelné.

## Fáze 4 — Portfolio, riziko a nižší obrat

- [ ] Zavést cílové váhy, limit jedné pozice a sektorové koncentrace.
- [ ] Přidat position sizing podle ATR/volatility a pravidla pro maximální riziko portfolia.
- [ ] Testovat MA200, trailing stop a 50denní low jako samostatné varianty, ne jako předem jisté zlepšení.
- [ ] Přidat hysterézi: obchod jen když je změna skóre významná a očekávaný přínos převyšuje náklady.
- [ ] Reportovat obrat, počet obchodů, dobu držení a příspěvek nákladů k výnosu.

## Fáze 5 — Robustnost a release 1.0

- [ ] Spustit citlivostní testy napříč parametry a obdobími trhu.
- [ ] Vyhodnotit výkonnost po sektorech, regionech, velikosti firmy a v krizových obdobích.
- [ ] Reportovat CAGR, volatilitu, Sharpe, maximální drawdown, obrat, expozice, benchmark a interval nejistoty.
- [ ] Přidat automatické testy výpočtů, testy úniku budoucích dat a smoke test Streamlit aplikace.
- [ ] Zdokumentovat datové licence, známá omezení a reprodukční postup.

**Podmínka finální verze:** žádné tvrzení o strategické výhodě bez out-of-sample důkazů; žádné automatické zadávání obchodů; veřejné UI zřetelně odděluje data, signály, simulaci a investiční doporučení (které neposkytuje).

## Pracovní pořadí

1. Fáze 0 — opravy výpočtů, automatické testy a ověření nasazení.
2. Fáze 1 — data a provenance.
3. Fáze 2 — point-in-time fundamenty.
4. Fáze 3 — realistický backtest a benchmarky.
5. Fáze 4 — řízení portfolia a obratu.
6. Fáze 5 — robustnost a release 1.0.

Nezačínat sentimentem ani HMM, dokud nejsou spolehlivá data, časová správnost a základní backtest. Tyto doplňky musí prokázat přínos na holdout období po nákladech.
