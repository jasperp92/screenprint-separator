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

Pantone-Bezeichnungen sind Metadaten. Für reproduzierbare Ergebnisse müssen
die zugehörigen LAB-Werte aus einer lizenzierten Quelle oder eigenen Messungen
eingetragen werden.

## Export

Der ZIP-Export enthält:

- die Simulation als PNG,
- drei binäre, Group-4-komprimierte 1-Bit-TIFF-Platten,
- `projekt.json` mit Druckreihenfolge, LAB-Palette und Einstellungen.

Die TIFF-Platten selbst sind absichtlich Schwarz-Weiß-Bitmaps. LAB-Werte werden
für Klassifikation und Simulation verwendet und im Projektmanifest gespeichert.
