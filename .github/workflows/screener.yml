name: Wheel-Screener
on:
  schedule:
    - cron: '0 15 * * 1-5'
  workflow_dispatch:
permissions:
  contents: write
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python wheel_screener.py
        env:
          EULERPOOL_API_KEY: ${{ secrets.EULERPOOL_API_KEY }}
      - run: |
          git config user.name wheel-bot
          git config user.email wheel-bot@users.noreply.github.com
          git add docs
          git commit -m "Screener $(date +%F)" || echo "Keine Aenderungen"
          git push
