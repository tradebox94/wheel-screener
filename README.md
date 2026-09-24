# Wheel-Screener

Zeigt jeden Handelstag die besten Cash-Secured Puts für deine Wheel-Strategie.

## Einrichtung (einmalig, ca. 10 Minuten)
1. Kostenloses Konto auf github.com anlegen.
2. Neues Repository erstellen (z. B. "wheel-screener", auf "Private" oder "Public").
3. Alle Dateien aus diesem Ordner hochladen ("Add file" > "Upload files"),
   inklusive des Ordners .github/workflows.
4. Settings > Pages: Source "Deploy from a branch", Branch "main", Ordner "/docs".
5. Settings > Secrets and variables > Actions > New repository secret:
   Name EULERPOOL_API_KEY, Wert = dein Eulerpool-API-Schlüssel.
6. Actions > "Wheel-Screener" > "Run workflow", um den ersten Scan zu starten.

Danach läuft der Scan Mo–Fr um 17:00 Uhr (MESZ) automatisch.
Die Ergebnisse siehst du unter https://DEINNAME.github.io/wheel-screener/
(als Lesezeichen aufs Handy legen).

Hinweis: GitHub Pages funktioniert bei privaten Repositories nur mit einem
kostenpflichtigen Plan. Bei "Public" ist alles kostenlos, die Ergebnisse
sind dann aber öffentlich sichtbar.

## Anpassen
- Kriterien: Block CFG oben in wheel_screener.py
- Eigene Aktienliste: watchlist.example.txt in watchlist.txt umbenennen
- Uhrzeit: cron-Zeile in .github/workflows/screener.yml (in UTC)

## Lokal starten (PC mit Python)
pip install -r requirements.txt
python wheel_screener.py
Ergebnis: docs/index.html
