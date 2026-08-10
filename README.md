# Screenprint Separator

NiceGUI-Anwendung zur Separation eines Bildes auf drei frei definierbare
Siebdruckfarben. Die Vorschau klassifiziert Bildfarben gegen acht mögliche
Druckzustände (Papier, drei Einzelfarben, drei Zweifachüberdrucke und den
Dreifachüberdruck).

## Start

```powershell
.\.venv\Scripts\screenprint-separator.exe
```

Anschließend ist die Oberfläche unter <http://localhost:8080> erreichbar.

## Farbverarbeitung

- CIELAB D50 für Eingabe- und Palettenfarben
- CIEDE2000 für die wahrnehmungsbezogene Farbdistanz
- schnelle quantisierte ΔE00-Lookup-Tabelle für die interaktive Vorschau
- exakte kachelweise ΔE00-Klassifikation für den Export
- frei wählbare Druckreihenfolge, Überdruckstärke und Bias je Farbe
- CMYK-Eingabe oder Auswahl aus einer lokalen JSON-Farbbibliothek

## CMYK und Pantone

Die drei Druckfarben werden in der Oberfläche entweder als CMYK-Prozentwerte
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

Papierfarben verwenden `"system": "paper"` und erscheinen ausschließlich im
Dropdown „Bedruckstoff / Papierfarbe“. Sie sind keine vierte Druckfarbe und
erzeugen keine Platte:

```json
{
  "id": "paper-natural-white",
  "name": "Naturweiß",
  "system": "paper",
  "rgb": [238, 233, 215]
}
```

Zusätzlich bietet das Dropdown „Eigene Papierfarbe …“. Diese Auswahl blendet
einen Farbwähler ein und verwendet die gewählte Farbe für den unbedruckten
Hintergrund, die Papier-Klasse und gegebenenfalls freie Ränder beim Einpassen.

## Export

Der ZIP-Export enthält:

- die Simulation als PNG,
- drei binäre, Group-4-komprimierte 1-Bit-TIFF-Platten,
- `projekt.json` mit Druckreihenfolge, LAB-Palette und Einstellungen.

Die TIFF-Platten selbst sind absichtlich Schwarz-Weiß-Bitmaps. LAB-Werte werden
für Klassifikation und Simulation verwendet und im Projektmanifest gespeichert.
