# Screenprint Separator

Mit dem Siebdruck-Separator lassen sich digitale Farbauszüge für
Belichtungsfilme (Filmpositive) vorbereiten. Das Programm berechnet und
simuliert das Überdrucken der Farben. Es können eine bis fünf verschiedene
Druckfarben ausgewählt und bis zu 32 Überdruckzustände simuliert werden. Die
exportierten Farbauszüge dienen als Vorlagen für die Belichtungsfilme, mit denen
die einzelnen Siebe belichtet werden.

Dies ist ein Vibecoding-Projekt von mir, das ich überwiegend mit ChatGPT Codex
(GPT-5.6 Sol, Reasoning-Stufe „High“) entwickelt habe.

## Technologien und Libraries

- **Python 3.13** als Programmiersprache und Laufzeit
- **NiceGUI** für die lokale, browserbasierte Benutzeroberfläche
- **NumPy** für Bilddaten, Masken und schnelle Pixeloperationen
- **Pillow** für Bildbearbeitung, Skalierung und Datei-Export
- **CIELAB D50 und CIEDE2000** für die wahrnehmungsbezogene Farbzuordnung
- **uv** für Abhängigkeiten, virtuelle Umgebung und Projektbefehle
- **PyInstaller / nicegui-pack** für eigenständige Windows- und macOS-Builds
- **Ruff** für statische Codeprüfung
- **GitHub Actions** für automatisierte Windows- und macOS-Builds mit Releases

## TODO / Nice to have

- Bildbearbeitung um weitere stapelbare Effekte und Presets ausbauen
- ICC-Profile und weitergehendes Farbmanagement ergänzen
- Undo/Redo sowie speicherbare Einstellungs- und Export-Presets hinzufügen
- Passermarken und weitere Hilfen für die Druckvorstufe integrieren

## Entwicklung

Voraussetzungen sind Python 3.13 und der Paketmanager
[`uv`](https://docs.astral.sh/uv/). Nach dem Klonen des Repositories installiert
folgender Befehl Python und alle in `pyproject.toml`/`uv.lock` festgelegten
Komponenten in die lokale virtuelle Umgebung:

```bash
uv sync --dev
```

Danach wird die NiceGUI-Anwendung im Entwicklungsmodus gestartet:

```bash
uv run screenprint-separator
```

Alternativ kann das Startmodul direkt ausgeführt werden:

```bash
uv run python src/screenprint_separator/app.py
```

Die Oberfläche ist anschließend unter <http://localhost:8080> erreichbar.
Änderungen an Python-Dateien unter `src/` lösen automatisch einen Hot Reload
aus. Der lokale Sitzungscache stellt dabei Einstellungen und Eingabebild wieder
her.

## Windows-EXE erstellen

Eine Windows-EXE muss unter Windows gebaut werden; PyInstaller erzeugt keine
Windows-Datei von macOS oder Linux aus. Im Projektverzeichnis werden zunächst
die regulären Abhängigkeiten installiert:

```powershell
uv sync --dev
```

Danach erstellt NiceGUIs PyInstaller-Hilfsprogramm eine einzelne ausführbare
Datei. `uv` installiert PyInstaller dafür vorübergehend, ohne die
Projektabhängigkeiten zu verändern:

```powershell
uv run --with pyinstaller nicegui-pack --name ScreenprintSeparator --onefile --clean --noconfirm --add-data "color_library.json;." src/screenprint_separator/app.py
```

Das Ergebnis befindet sich anschließend hier:

```powershell
.\dist\ScreenprintSeparator.exe
```

Die EXE startet den lokalen Server und öffnet die Oberfläche im Browser. Im
Build wird Hot Reload automatisch deaktiviert. `color_library.json` wird durch
`--add-data` in die EXE aufgenommen. Der Sitzungscache liegt bei der gebündelten
Anwendung dauerhaft unter
`%USERPROFILE%\.screenprint_separator\.screenprint_separator_cache\`.
Der App-Einstiegspunkt initialisiert PyInstallers Multiprocessing-Unterstützung.
In der gebündelten Windows-EXE läuft der Export speicherschonend in einem
Hintergrund-Thread, sodass kein zweiter vollständiger EXE-Prozess gestartet und
keine zusätzliche Kopie des hochauflösenden Bilds übertragen werden muss.

## macOS-App erstellen

Eine macOS-App muss auf einem Mac gebaut werden. PyInstaller erzeugt dabei eine
App für die Architektur des Build-Rechners, also Apple Silicon oder Intel.
Zunächst werden die Projektabhängigkeiten installiert:

```bash
uv sync --dev
```

Danach erstellt NiceGUIs Packaging-Hilfe eine eigenständige `.app`. PyInstaller
und `pywebview` werden von `uv` nur für diesen Build bereitgestellt:

```bash
uv run --with pyinstaller --with pywebview nicegui-pack \
  --name ScreenprintSeparator \
  --windowed \
  --onedir \
  --clean \
  --noconfirm \
  --add-data "color_library.json:." \
  src/screenprint_separator/app.py
```

Die fertige Anwendung liegt anschließend hier:

```text
dist/ScreenprintSeparator.app
```

Sie kann im Finder per Doppelklick oder im Terminal gestartet werden:

```bash
open dist/ScreenprintSeparator.app
```

Der macOS-Build öffnet die Oberfläche als natives App-Fenster mit `pywebview`.
Hot Reload ist darin deaktiviert. Für die lokale Entwicklung bleibt weiterhin
der Browsermodus aktiv. Exporte aus der gebündelten App werden direkt unter
`~/Downloads` gespeichert; vorhandene Dateien werden dabei nicht überschrieben.
Da ein lokaler Build nicht mit einem Apple Developer
Certificate signiert oder notarisiert ist, kann macOS beim ersten Start warnen.
In diesem Fall kann die App im Finder über Rechtsklick → „Öffnen“ bestätigt
werden. Für die Verteilung an andere Macs sollte die App mit einer Developer-ID
signiert und anschließend von Apple notarisiert werden.

Der Sitzungscache der App befindet sich dauerhaft unter:

```text
~/.screenprint_separator/.screenprint_separator_cache/
```

### Automatischer Build mit GitHub Actions

Der Workflow `.github/workflows/build-macos.yml` kann unter GitHub → Actions →
„Build Desktop Apps“ manuell gestartet werden. Die macOS-App und die
Windows-EXE stehen danach als separate Workflow-Artefakte zum Download bereit.

Wird ein Versions-Tag gepusht, erstellt der Workflow zusätzlich automatisch ein
GitHub Release und hängt beide gebauten Anwendungen daran:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Der macOS-Dateiname enthält die Architektur des GitHub-Runners, beispielsweise
`ScreenprintSeparator-macOS-arm64.zip`. Der Windows-Build wird als
`ScreenprintSeparator.exe` veröffentlicht. Der Workflow erzeugt zunächst
unsignierte Builds. Für eine öffentliche macOS-Verteilung müssen später
Developer-ID-Zertifikat und Notarisierungsdaten als GitHub-Secrets ergänzt
werden.

## Effekte

Über das Plus im Bereich „Effekte“ können beliebig viele Bildkorrekturen
hinzugefügt werden. Die Karten lassen sich per Drag & Drop sortieren und werden
von oben nach unten auf das Eingabebild angewandt. Verfügbar sind derzeit:

- Sättigung / Dynamik
- Helligkeit / Kontrast
- Schwarzweiß mit einstellbarer Stärke
- Tonwertkorrektur mit RGB-Histogramm, Schwarzpunkt, Gamma und Weißpunkt
- Selektive Farbe mit CIELAB-Zielfarbe, Toleranz und weicher Auswahl
- Farbtonverschiebung von −180° bis +180° mit zyklischer Farbvorschau

Die Effekte sind nicht-destruktiv, werden in der Sitzung gespeichert und gelten
gleichermaßen für Vorschau und Export. „Textur und Glättung“ bleibt ein eigener
Verarbeitungsschritt und wird nach dem Effektstapel angewandt.

Die Pipetten an Papier-, Druck- und selektiver Zielfarbe zeigen beim Überfahren
des Eingabebilds eine Live-Farbvorschau. Der Messradius lässt sich platzsparend
im Pipetten-Menü von einem Einzelpixel bis zu einer gemittelten Umgebung wählen.

## Farbverarbeitung

- CIELAB D50 für Eingabe- und Palettenfarben
- CIEDE2000 für die wahrnehmungsbezogene Farbdistanz
- schnelle quantisierte ΔE00-Lookup-Tabelle für die interaktive Vorschau
- exakte kachelweise ΔE00-Klassifikation für den Export
- frei wählbare Druckreihenfolge, Überdruckstärke und Bias je Farbe
- automatisch per RGB-Alphamischung angenäherte oder manuell referenzierte
  Mischfarben für alle Überdruckkombinationen
- CMYK-Eingabe oder Auswahl aus einer lokalen JSON-Farbbibliothek
- geordneter, nicht-destruktiver Effektstapel für das Eingabebild

## AM-Raster und Verläufe

Unter „Eingabe → Rasterung“, direkt unter „Textur und Glättung“, kann zwischen
der bisherigen Volltontrennung und einem AM-Raster gewählt werden. Im
Rastermodus berechnet die Farbzuordnung zunächst kontinuierliche
Flächendeckungen aus den CIEDE2000-Abständen zu allen Vollton- und
Überdruckzuständen. Erst danach werden daraus binäre Rasterpunkte erzeugt.
Dadurch bleiben weiche Farb- und Tonwertverläufe in den 1-Bit-Farbauszügen
erhalten.

Einstellbar sind:

- Rasterweite in lpi
- Punktform Kreis, Ellipse oder Kreuz
- distanzgewichtete Verlaufsweichheit
- Tonwertkurve sowie Minimal- und Maximalpunkt
- ein eigener Rasterwinkel je Druckfarbe

Die interaktive Simulation rastert in der aktuellen Vorschauauflösung. Für die
TIFF-Farbauszüge werden die kontinuierlichen Flächendeckungen dagegen zuerst auf
die Ausgabegröße skaliert und anschließend direkt bei den eingestellten
Ausgabe-DPI gerastert. So bleiben Rasterweite und Winkel physisch definiert.
Automatisch und manuell bestimmte Überdruckfarben werden auch für die
Rastervorschau verwendet. Trapping ist im Rastermodus bewusst deaktiviert, da
eine nachträgliche Maskenerweiterung dort einem unkontrollierten Punktzuwachs
entsprechen würde.

## CMYK und Pantone

Die ein bis fünf Druckfarben werden in der Oberfläche entweder als CMYK-Prozentwerte
eingegeben oder aus `color_library.json` ausgewählt. RGB und LAB werden intern
berechnet und müssen nicht manuell bedient werden.

Die generische CMYK-Umrechnung ist eine Vorschau ohne konkretes ICC-Druckprofil.
Für verbindlichere Pantone-Ergebnisse kann die lokale Farbbibliothek LAB-Werte
aus Pantone Connect, einer anderen lizenzierten Quelle oder eigenen Messungen
enthalten. Beispiel:

```json
{
  "id": "meine-lizenzierte-farbe",
  "name": "PANTONE-Bezeichnung",
  "system": "pantone",
  "cmyk": [10, 80, 40, 5],
  "lab": [52.3, 48.1, 12.7],
  "rgb": [190, 62, 92]
}
```

`lab` hat für die Klassifikation Vorrang. `rgb` wird nur für die
Bildschirmsimulation benutzt. Fehlt `rgb`, wird es aus LAB abgeleitet; fehlen
LAB und RGB, werden beide näherungsweise aus CMYK berechnet. Nach Änderungen an
der JSON-Datei kann sie über „Farbbibliothek neu laden“ aktualisiert werden.

Papierfarben verwenden `"system": "paper"` und erscheinen ausschließlich in
der Papierauswahl. Sie zählen nicht als Druckfarbe und erzeugen keinen
Farbauszug:

```json
{
  "id": "paper-natural-white",
  "name": "Naturweiß",
  "system": "paper",
  "rgb": [238, 233, 215]
}
```

Alternativ kann die Papierfarbe über CMYK-Werte oder den dazugehörigen
Farbwähler gewählt oder als eigene LAB-Referenzfarbe eingegeben werden. Der
Farbwähler verwendet technisch sRGB und rechnet die Auswahl – wie bei den
Druckfarben – in CMYK um. Die Papierfarbe wird für den unbedruckten Hintergrund,
die Papier-Klasse und gegebenenfalls freie Ränder beim Einpassen verwendet.

## Mischfarben

Im Modus „Automatisch“ werden alle möglichen Überdruckkombinationen im RGB-Raum
angenähert. Die Reihenfolge der Druckfarben legt dabei fest, welche Farbe oben
liegt; deren „Überdruckstärke“ ist der Alpha-Wert.

Für jede Überdruckkombination kann die Näherung durch eine Pantone-Farbe aus der
JSON-Bibliothek oder einen gemessenen CIELAB-D50-Wert ersetzt werden. Diese Werte
gelten sowohl für Klassifikation als auch für Simulation und Export. Beim
Umsortieren, Ergänzen oder Löschen bleiben Referenzen ihrer jeweiligen
Farbkombination zugeordnet.

## Export

Die Export-Einpassung wird direkt in der Separationsvorschau dargestellt. Bei
„Format füllen“ und „Freier Rahmen“ bleibt der exportierte Bereich hell, während
abgeschnittene Bildteile leicht abgedunkelt werden. „Einpassen mit Rand“ zeigt
das vollständige Bild einschließlich des ergänzten Papierbereichs.

Der Rahmen verhält sich ähnlich wie das Freistellungswerkzeug in Photoshop:
Ziehen innerhalb des Rahmens verschiebt ihn, Ziehen an einer Kante verändert
diese Seite und Ziehen an einer Ecke skaliert ihn in beide Richtungen. Bei
„Format füllen“ bleibt dabei das Seitenverhältnis des Druckformats gesperrt. Im
Modus „Freier Rahmen“ darf es sich ändern und bestimmt dann automatisch das
Seitenverhältnis des Druckformats und der exportierten Farbauszüge. Der
Ausschnitt wird zusammen mit der Sitzung gespeichert. Rahmenlinien, Raster und
Griffe werden nur während der aktiven Bearbeitung eingeblendet; die
abgedunkelte Beschnittmaske bleibt zur Exportkontrolle sichtbar.

Beim Auswählen von „Format füllen“ sowie nach Änderungen an Druckformat oder DPI
wird zunächst wieder der größtmögliche Ausschnitt im festen Seitenverhältnis
verwendet. Danach kann der Rahmen erneut verschoben oder verkleinert werden.

Der ZIP-Export enthält:

- die Simulation als PNG,
- eine bis fünf binäre, Group-4-komprimierte 1-Bit-TIFF-Farbauszüge,
- `projekt.json` mit Druckreihenfolge, LAB-Palette und Einstellungen.

Die TIFF-Farbauszüge selbst sind absichtlich Schwarz-Weiß-Bitmaps. LAB-Werte
werden für Klassifikation und Simulation verwendet und im Projektmanifest
gespeichert.

## Sitzungswiederherstellung

Die Anwendung speichert die letzten Einstellungen und das zuletzt geladene Bild
lokal unter `.screenprint_separator_cache/`. Nach einem Neustart oder Hot Reload
werden Farben, Reihenfolge, Regler, Überdruckquellen und Eingabebild automatisch
wiederhergestellt. Der Cache ist über `.gitignore` vom Repository ausgeschlossen.

### Papier als feste Grundfarbe

Papier ist immer die erste Farbe mit 100 % Deckkraft. Jede Druckfarbe wird
mit ihrer Überdruckstärke auf die darunterliegende Farbe gemischt, auch die
erste: 50 % Schwarz auf weißem Papier ergibt Grau. Weitere Druckfarben werden
in Druckreihenfolge darübergelegt. Automatische LAB-Werte werden aus diesen
Mischfarben berechnet; explizite Pantone- oder LAB-Messwerte eines Zustands
ersetzen dessen automatische Farbe.

Es gibt genau einen Zustand je Druckplattenkombination. Die Palette unter dem
Bild ist nach Leveln geordnet: Level 1 = Papier, Level 2 = Papier + eine
Druckfarbe, Level 3 = Papier + zwei Druckfarben usw. Bei fünf Druckfarben sind
es 32 Zustände und fünf Druckplatten. Dieselbe Palette gilt für Vollton,
Raster, Überfüllung, Vorschau und Export.

Die Papierkarte bietet Hintergrundfarben aus der Bibliothek, CMYK, Farbwähler,
Pipette, Farbvorschau, LAB-Referenzfarbe und Klassifikations-Bias. Positive
Bias-Werte bevorzugen unbedruckte Papierflächen. Papier benötigt weder
Überdruckstärke noch Rasterwinkel oder eine eigene Druckplatte. Der bisherige
Schalter entfällt, da Papier immer die Basis ist. Gespeicherte separate
Papiermischungen werden beim Laden auf die entsprechenden Zustände übertragen.

Mischungen mit mindestens zwei Druckfarben besitzen einen zusätzlichen
„Mischfarben-Bias (ΔE)“. Ein positiver Wert bevorzugt genau diesen Zustand,
ein negativer Wert reduziert ihn. Beispielsweise verstärkt ein Bias für
„Papier + Rot + Gelb“ die Orange-Mischung, ohne die Gewichtung von Weiß oder
von „Papier + Gelb + Weiß“ anzuheben. Der Wert wird zum Bias der beteiligten
Druckfarben addiert und wirkt in Vollton- und Rastermodus, Vorschau und Export.
Er wird gespeichert und bleibt beim Umordnen der Druckfarben der gleichen
Kombination zugeordnet. Der Ausgangswert ist 0.

Die Primärdruckfarben werden ausschließlich in ihren Druckfarbenkarten
bearbeitet. Ihre Farbfelder zeigen direkt die mit Papier und Überdruckstärke
berechnete Farbe; Farbwähler und CMYK-Eingaben definieren weiterhin die
Ausgangsfarbe. Unter Überdruckfarben erscheinen nur Kombinationen aus mindestens
zwei Druckfarben. In der Legende unter dem Bild trennen Linien die Gruppen;
zusätzliche Level-Beschriftungen entfallen.

Im Rastermodus werden neutrale Töne über einen neutralen Rasterverlauf
wiedergegeben, sofern die Palette zwei neutrale Zustände bietet, die sich in
genau einer Druckplatte unterscheiden. Auf schwarzem Papier mit weißer Farbe
entstehen Graustufen dadurch ausschließlich aus weißen Rasterpunkten. Die
Deckung folgt dem Flächenmittel der beiden simulierten Endfarben; hellere
Vorlagenwerte als das verfügbare Weiß werden auf volle Weißdeckung begrenzt.
Die Neutralbehandlung gilt vollständig bis LAB-Chroma 4 und läuft bis Chroma 12
weich aus. Gesättigte Farben behalten die bisherige Mischfarbenzuordnung.
Bias kann den neutralen Ton verschieben, führt dort aber keine bunten Platten
hinzu. Gamma und Minimalpunkt wirken weiterhin anschließend auf die Deckung.

Die Oberfläche bezeichnet Papier jetzt als „Hintergrund“. Dessen Bibliotheksliste
enthält ausschließlich Hintergrundfarben, einschließlich „Schwarz“ (#000000).
Druckfarbentitel lassen sich per Klick direkt bearbeiten: Enter oder Verlassen
des Felds übernimmt den Namen, Escape bricht ab. Eigene Namen bleiben bei
Farbänderungen und nach einem Neustart erhalten und erscheinen auch bei
Druckreihenfolge, Druckplatten und Export.

Der Hintergrund trägt in Karten, Kombinationen und Legende die Nummer 0
(zum Beispiel „0 + 1 + 2“). Über seiner Karte steht einmal „Hintergrundfarbe“.
