# Campus Facility Reservation — Data Mining Lab Exam (Streamlit)

**Course:** Systems Analysis and Design (SAD) · **Program:** BS Information Systems
**Student:** Christian P. Tuyor · **Section:** BSIS-3A

Streamlit version of the Colab notebook *Data Mining and Systems Analysis Integration*.
Pipeline: historical data → cleaning → analysis → prediction → requirements → use cases → decision support.
All data are **synthetic** — no real personal information.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Folder structure

```
facility_streamlit/
├── app.py                  # Streamlit UI: tabs for Parts I–VI + downloads, EDA, model, dashboard
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # theme / server settings
├── data/                   # sample files generated with the default seed (42)
│   ├── facility_reservations.csv          # raw, deliberately messy
│   └── facility_reservations_clean.csv    # after Part II cleaning
└── src/
    ├── data_generator.py   # Section 0: synthetic dataset with injected data-quality problems
    ├── cleaning.py         # Tasks 4–5: category maps, validation ranges, derived variables
    └── diagram.py          # Task 10: use-case diagram (matplotlib preview + draw.io XML)
```

## Using the app

* **Sidebar** — generate synthetic data (seed, 300–500 rows) or upload your own
  `facility_reservations.csv` (e.g. `data/facility_reservations.csv`); tune the Decision Tree
  (`max_depth`, `min_samples_leaf`, test share) and the Low/High risk thresholds.
* **Tabs** — Overview · I Understanding · II Cleaning · III EDA · IV Modelling · V Requirements ·
  VI Decision support · Downloads.
* **Downloads** — raw/clean CSVs, the `.drawio` use-case diagram (open at https://app.diagrams.net),
  the diagram PNG and the dashboard PNG.

## Deploy (Streamlit Community Cloud)

Push this folder to GitHub, then create an app pointing at `app.py`. No secrets needed.
