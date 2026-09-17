# OpenEuroLLM Token Counts

This repository contains the token counts and language distribution analysis for the total flagship mix trained in OpenEuroLLM. 

## Overview

The analysis extracts language information from the `part` column in the `counts.csv` dataset, aggregates the token budget, and computes the percentages of the total tokens for each language.

### Key Findings
- A significant portion of the tokens (over 69%) are categorized under "Other", primarily due to large generic datasets (e.g., `common-pile`, `nemotron-cc`) that do not specify a language tag in the `part` column.
- **English** (`eng_Latn`) is the most prominent explicitly tagged language, followed by **Spanish** (`spa_Latn`), **French** (`fra_Latn`), and **German** (`deu_Latn`).
- **Math** and **Code** make up a measurable portion of the specifically tagged data.

## Visualizations

### Bar Chart
![Tokens by Language Bar Chart](assets/bar_chart.png)

### Pie Chart
*(Languages with <1.5% of the total tokens are grouped into "Other < 1.5%")*
![Tokens by Language Pie Chart](assets/pie_chart.png)

## Running the Analysis

The analysis script uses `polars` for fast data processing and `seaborn`/`matplotlib` for visualization.

To run the script:

```bash
# Ensure dependencies are installed (e.g. using uv)
uv sync

# Run the analysis
uv run python src/analyze.py
```

## Testing

A `pytest` suite is included to verify the language extraction logic. Run tests with:

```bash
PYTHONPATH=. uv run pytest tests/
```
