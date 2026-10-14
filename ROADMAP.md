# Quality × Value × Momentum Lab — roadmap (research app v2.6.0)

## Finální cíl

Reprodukovatelná veřejná výzkumná aplikace pro screening akcií a testování strategie s nízkou obrátkou. Každá metrika má zdroj, datum a přesné pravidlo výpočtu; historická rozhodnutí používají pouze tehdy dostupné informace; výsledky ukazují obrat, riziko, náklady a porovnání s benchmarkem. Dokud neprojde walk-forward a odloženým out-of-sample testem, aplikace nesmí tvrdit, že strategie má prokázanou výhodu.

## Fáze 0 — Oprava výpočtů a základní QA

- [x] Opravit časování signálu a návratnosti dalšího měsíce v backtestu.
- [x] Počítat obrat jako hrubý zobchodovaný objem (nákupy + prodeje).
- [x] Počítat průraz z předchozích 20/55 denních high a ATR(20) z true range.
- [x] Zachovat chybějící faktory jako chybějící; nevyplňovat je fiktivním neutrálním skóre 50.
- [x] Přidat regresní testy indikátorů, časování, look-ahead, SEC as-of date, vah a FIFO.
- [x] Přidat GitHub Actions test workflow.
- [ ] Ověřit aktuální živou verzi na Streamlit Community Cloud a zkontrolovat log po posledních změnách.

## Fáze 1 — Data a provenance

- [x] Oddělit cenový adapter od fundamentálního zdroje; přestat používat nestabilní Yahoo `.info` endpoint pro fundamenty.
- [x] Zaznamenat měnu každého OHLCV tickeru, odmítnout nekonzistentní měnu v řadě a zablokovat portfolio hodnoty, backtest a sizing při známé ne-USD nebo smíšené měně, protože FX feed není implementován.
- [x] Stahovat ceny v jedné neparalelní dávce, normalizovat různé tvary sloupců a při částečném selhání opakovat chybějící tickery sekvenčně.
- [x] Vykazovat první/poslední datum cenové řady, počet řádků, chybějící řady a exportovat kontrolní CSV.
- [x] Přidat oficiální SEC EDGAR Company Facts adaptér s identifikačním User-Agent, SIC kódem/popisem a opakováním při přechodných HTTP chybách.
- [x] Detekovat zastaralé řady, neplatné OHLC, extrémní denní pohyby a zobrazovat stav `OK/REVIEW` v datovém panelu; neplatné cenové řádky se z výpočtů vyřazují.
- [x] Použít adjustované OHLC pro MA/ATR/breakout/denní stop, aby historické splitové úpravy nevytvářely falešné cenové skoky.
- [ ] Porovnat konkrétní tickery s nezávislým cenovým zdrojem a ručně zkontrolovat vybrané XBRL metriky.
- [ ] Doplnit automatickou kontrolu duplicit po transformaci, corporate actions a varování na příliš řídkou historii.
- [ ] Vybrat zdroj s vhodnými právy pro veřejné použití a redistribuci; Yahoo/yfinance zůstává pouze zdrojem pro experimentální výzkum, dokud nejsou práva vyřešena.

**Brána fáze:** žádný chybějící nebo starý údaj nesmí být tiše prezentován jako ověřená aktuální data.

## Fáze 2 — Fundamenty a časová správnost

- [x] Zavést SEC snapshoty filtrované podle data zveřejnění a konzervativně odložit fakta zveřejněná v den uzavíracího signálu až na další rozhodovací den.
- [x] Vypočítat roční růst tržeb/zisku, ROE proxy, marži, FCF a FCF marži z dostupných standardizovaných výkazů.
- [x] Vykazovat fundamentální pokrytí a stav SEC dat; chybějící metriky zůstávají prázdné.
- [x] Přidat samostatný FCFE-style DCF scénář s editovatelnými předpoklady a citlivostní tabulkou, oddělený od Composite score a obchodního backtestu.
- [ ] Doložit měnové/jednotkové mapování FCF, počet akcií a quote price pro emitenta/ADR/dual-class před použitím DCF výsledku.
- [x] Přidat časované CSV vstupy pro sentiment a analytické revize; historie používá poslední záznam známý k rozhodovacímu datu.
- [ ] Ověřit SEC tagy, jednotky, fiskální roky, opravy výkazů, ADR a duální třídy pro reprezentativní vzorek.
- [ ] Doplnit point-in-time mapování historie tickerů/CIK, segmentové metriky, sektorově srovnatelné metriky a transparentní zacházení se záporným vlastním kapitálem.
- [ ] Získat dlouhodobou historii analytických revizí a sentimentu s doložitelným časovým razítkem a právy k použití; dokud není k dispozici, nesmí být skóre vyplňováno odhadem.

**Brána fáze:** historické rozhodnutí nevyužívá informaci, která nebyla veřejně dostupná v daném okamžiku. SEC prototyp pokrývá jen část universe a není ještě účetně ověřeným komerčním datovým feedem.

## Fáze 3 — Backtest a benchmarky

- [x] Měsíční signál na konci měsíce, výnos za následující měsíc a audit vybraných pozic.
- [x] Přidat měsíční obrat, náklady v bps, CAGR, volatilitu, Sharpe, drawdown, SPY benchmark a export equity curve/auditu.
- [x] Přidat hysterézi, limit výměn na rebalancování a měsíční výstupní spouštěč 50denního minima.
- [x] Přidat volbu Equal Weight vs. Inverse ATR se stropem váhy, neinvestovaný zbytek zůstává v hotovosti.
- [x] Přidat volitelné výstupy: bez stopu, měsíční 50denní low, denní 50denní low, ATR trailing stop nebo obojí; denní verze zaznamenává stop a datum výstupu a používá konzervativní přibližné plnění z denního OHLC.
- [x] Přidat citlivost CAGR/volatility/Sharpe/drawdown na různé náklady při zachování stejné historie signálů.
- [x] Přidat 60/20/20 chronologické diagnostické rozdělení Train / Validation / Holdout včetně benchmarku (není to automatizovaný walk-forward optimizer; holdout zůstává nedotčený jen tehdy, pokud se parametry podle něj neupravují).
- [x] Přidat reprodukovatelný moving-block bootstrap pro orientační intervaly CAGR, Sharpe, max drawdown a konečné hodnoty; intervaly jsou diagnostické, nikoli garance statistického pokrytí.
- [x] Přidat lokální grid min-score × hystereze, vybrat konfiguraci jen z tréninkového Sharpe a reportovat pozdější holdout pouze pro vítěznou konfiguraci; nejde o rolling walk-forward a opakované ladění podle holdoutu jej znehodnotí.
- [x] Přidat intervalový CSV import historického členství (Ticker/Start date/End date), validovat překryvy a chybějící OHLCV řady a filtrovat členy před cross-sectional percentile rankingem.
- [ ] Získat nebo nezávisle ověřit úplný, licencovaný point-in-time security master včetně delistovaných titulů, symbolických změn a historických změn složení; uživatelský CSV import sám o sobě úplnost nedokazuje.
- [x] Rozložit náklady na komisi, polovinu bid/ask spreadu a skluz v bps na hrubý zobchodovaný objem, včetně modelových prodejních nákladů při stop výstupu.
- [ ] Nahradit hrubý cost proxy realistickým modelem exekuce: přesné open/close fill, likvidita/market impact, fixní poplatky, FX, dividendy a corporate actions.
- [x] Přidat expanding-window walk-forward: v každém foldu volit parametry pouze z minulých výnosů a reportovat jen vítěznou konfiguraci v neprolínajícím se OOS období. Každé OOS okno začíná s hotovostí a na konci účtuje modelovou likvidaci; nejde o nepřerušený broker účet.
- [ ] Rozšířit walk-forward o skutečný kontinuální přenos pozic mezi foldech, širší sadu předem registrovaných modelů a následně další benchmarky napříč regiony a režimové expozice.
- [ ] Prokázat, že chybějící/delistované výnosy nezkreslují výsledek; aktuální seznam tickerů stále vytváří survivorship bias.

**Brána fáze:** výsledek musí být reprodukovatelný, s náklady, bez look-ahead bias a s benchmarkem; používání aktuálního tickerového seznamu stále znamená survivorship bias.

## Fáze 4 — Portfolio, řízení rizika a daně

- [x] Přidat samostatný ATR risk-sizing kalkulátor s max. váhou pozice a maximální expozicí.
- [x] Přidat Inverse ATR vážení do backtestu, váhový strop a report skutečné expozice včetně hotovosti.
- [x] Přidat hysterézi a omezený počet náhradních pozic při měsíčním rebalancování.
- [x] Přidat FIFO evidence nákupů/prodejů s rozdělením poplatků, realizovaným výsledkem a otevřenými loty.
- [x] Přidat předběžné daňové kontrolní štítky pro české osvobození: roční hodnotový test do 100 000 Kč; 40milionový strop časového testu pouze pro rok 2025; odstranění tohoto stropu pro cenné papíry od roku 2026; kontrola potenciální oznamovací hranice 5 mil. Kč po zdrojovém prodejním řádku.
- [x] FIFO odmítá párování nákupního lotu a prodeje v různých měnách bez FX konverze a souhrnná KPI nesčítají P/L v různých měnách.
- [x] FIFO odmítá NaN/nekonečné hodnoty a chybějící měnu nikdy tiše nepřekládá na USD; neznámou měnu zachovává jako UNKNOWN a blokuje z ní závěry o CZK osvobození.
- [ ] Pro daňové účely doplnit FX do CZK, všechny prodeje a ostatní osvobozené příjmy za rok včetně kryptoaktiv (zejména pro rok 2025), corporate actions, dědictví a výjimky (např. obchodní majetek); současný modul je pouze orientační a není finálním výpočtem daně.
- [x] Přidat omezení počtu pozic podle současné hrubé SEC SIC skupiny, report skupinové expozice, počty vstupů/výstupů, průměrnou dobu držení a nákladové komponenty.
- [x] Přidat portfolio-level souhrnný risk budget do ATR sizing kalkulátoru a škálovat celé kusy pod cap.
- [ ] Získat point-in-time historické klasifikace a nahradit hrubé SIC skupiny srovnatelnou historickou sektorovou klasifikací; současné SIC metadata nejsou GICS/ICB a mohou se měnit.
- [x] Umožnit porovnání variant 50denního minima a ATR trailing stopu jako samostatných nastavení, ne jako předem jistého zlepšení.
- [ ] Ověřit limity na širším portfoliu a doplnit sektorové limity s historickou point-in-time klasifikací.

**Daňové upozornění:** modul zatím neurčuje daňovou povinnost, sazbu ani obsah přiznání. Pro orientaci vychází z veřejného shrnutí Finanční správy: https://financnisprava.gov.cz/cs/financni-sprava/media-a-verejnost/tiskove-zpravy-gfr/tiskove-zpravy-2026/danove-novinky-pro-rok-2026. U konkrétních případů je třeba ověřit zákonné podmínky a úplnost celého portfolia.

## Fáze 5 — Robustnost a release 1.0

- [x] Přidat automatické regresní testy indikátorů, chybějících skóre, chronologie a FIFO lotů.
- [x] Přidat CI smoke test kompletního Streamlit UI proti simulovaným cenovým datům bez živého cenového API.
- [x] Přidat týdenní snapshot review: export úplného screeneru a porovnání minulého snapshotu, prahů skóre/momenta, trendu, 50denního low, signálů a filing date s exportovatelnou frontou pro lidskou kontrolu.
- [ ] Ověřit poslední nasazený build, stavy všech GitHub Actions a reálný log Streamlit Cloud.
- [ ] Spustit parametrickou citlivost, walk-forward a zcela nedotčený holdout.
- [x] Reportovat nejistotu pořadí měsíčních výnosů pomocí moving-block bootstrapu společně s volatilitou, drawdownem, obratem, benchmarkem a expozicemi.
- [ ] Odhadnout statistickou významnost výhody při vícenásobném testování parametrů (deflated Sharpe / reality-check přístup) až s dostatečným bias-free datasetem.
- [ ] Dokumentovat licence zdrojů, limity pokrytí, reprodukční postup a povinné varování veřejného UI.
- [x] Dokumentovat snapshot review workflow; uživatel exportuje minulý CSV snapshot a v dalším týdnu jej ručně nahraje. Bez persistentního úložiště se historie mezi návštěvami automaticky neukládá.

**Podmínka release 1.0:** žádné tvrzení o strategické výhodě bez out-of-sample důkazu; žádné automatické obchodní pokyny; jasné oddělení zdrojových dat, skóre, simulace a osobního rozhodnutí.

## Pracovní pořadí

1. Ověřit poslední nasazení a projít živý log bez nových chyb.
2. Uzavřít kontroly zdrojů a SEC metrik na konkrétních tickerech.
3. Rozšířit point-in-time univerzum a přesnost historických dat.
4. Zrealističtit exekuce, náklady a portfolio risk.
5. Teprve poté spustit walk-forward a odložený holdout; sentiment/HMM ponechat jen pokud prokážou přírůstkovou hodnotu po nákladech.

HMM zůstává odložen: bez silného a dobře testovaného základního modelu by zvyšoval riziko přeoptimalizace a přidával složitost bez prokázaného přínosu.
