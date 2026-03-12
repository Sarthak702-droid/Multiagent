# NeuralGrid — Smart Building Multi-Agent Intelligence System

## Stack
- **Backend**: Flask REST API (Python)
- **Database**: SQLite with field-level AES-256 encryption
- **ML**: Random Forest · Gradient Boosting · MLP Neural Network
- **Frontend**: Vanilla JS + Chart.js — no framework required
- **Security**: Encrypted sensitive fields, IP hashing in audit log, row-level hashing

## Quick Start

```bash
pip install flask numpy pandas scikit-learn cryptography

cd smartbuilding
python app.py
# Open: http://localhost:5000
```

## Usage Flow

1. **Initialize & Train** — Generates 32,000 samples, trains 3 ML models
2. **Run Simulation** — 500-step multi-agent negotiation simulation
3. **Dashboard** — Live charts, fault timelines, agent performance
4. **ML Models** — Confusion matrix, model comparison, metrics
5. **Live Feed** — Real-time fault recovery stream with filtering
6. **Dataset** — Browse, filter, download (CSV/JSON)
7. **Predict** — Real-time single-reading fault prediction
8. **AI Report** — Multilingual predictive report (EN/HI/OR/ZH/DE)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/pipeline/init` | Generate dataset + train models |
| POST | `/api/pipeline/simulate` | Run multi-agent simulation |
| GET | `/api/status` | System status + progress |
| GET | `/api/dashboard` | Aggregated metrics |
| GET | `/api/ml/results` | Model performance + confusion matrices |
| GET | `/api/dataset/page` | Paginated dataset browser |
| GET | `/api/dataset/stats` | Dataset statistics |
| POST | `/api/predict` | Single-reading fault prediction |
| POST | `/api/ai_report` | Multilingual AI report (lang: en/hi/or/zh/de) |
| GET | `/api/download/dataset` | Download full dataset (CSV/JSON) |
| GET | `/api/download/recovery_log` | Download simulation recovery log |
| GET | `/api/live/events` | Latest 20 fault events |
| GET | `/api/runs` | All simulation runs |
| GET | `/api/runs/<id>` | Single run details + events |
| GET | `/api/analytics/fault_timeline` | Timeline data for charts |
| GET | `/api/analytics/recovery_dist` | Agent recovery distribution |
| GET | `/api/analytics/hourly_pattern` | 24h HVAC/occupancy pattern |

## Security Features

- **AES-256 encryption** (Fernet) on sensitive run summaries
- **IP hashing** (SHA-256) — raw IPs never stored
- **Row deduplication** via SHA-256 hashing on insert
- **Audit log** for all critical actions (pipeline, downloads, predictions)
- Input validation on all prediction endpoints

## Project Structure

```
smartbuilding/
├── app.py           # Flask REST API
├── core.py          # ML models + agent framework
├── database.py      # SQLite + encryption layer
├── requirements.txt
├── data/
│   ├── smartbuilding.db   # SQLite database (auto-created)
│   └── .enc_key           # Fernet encryption key (auto-generated)
└── templates/
    └── index.html    # Full frontend (single file)
```
