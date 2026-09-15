# OralScan — BEiT + SHAP-selected features + Random Forest

This package connects the OralScan GitHub Pages frontend to the saved research model.

## Architecture

Browser → FastAPI `/predict` → BEiT (`microsoft/beit-base-patch16-224`) → 768-D CLS token → the same 200 saved SHAP-ranked feature positions → saved 500-tree Random Forest → Normal/OSCC + confidence.

The frontend keeps the original OralScan design. The old random/mock prediction has been removed.

## Project structure

```text
oralscan_final/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── model/
│   ├── final_random_forest.pkl
│   ├── final_selected_features.npy
│   └── model_metadata.json
└── README.md
```

## Run locally

From this folder:

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

The first startup downloads the pretrained BEiT checkpoint from Hugging Face. The backend then stays loaded in memory.

Open `frontend/index.html` with a local static server, or use VS Code Live Server. The frontend defaults to `http://127.0.0.1:8000`.

Check:

```text
http://127.0.0.1:8000/health
```

## Publish the frontend on GitHub Pages

Keep `index.html`, `style.css`, and `script.js` in the GitHub Pages repository.

After deploying the backend, change this line near the top of `frontend/script.js`:

```javascript
const API_BASE_URL = 'https://YOUR-BACKEND-URL';
```

Then commit/push the frontend files.

## Backend deployment

The included Dockerfile is ready for a Docker-capable Python host. Set:

```text
ALLOWED_ORIGINS=https://shubham132328.github.io
```

If your GitHub Pages address is different, use that exact origin instead.

## Important research/model details

- Classification: Normal vs OSCC
- BEiT checkpoint: `microsoft/beit-base-patch16-224`
- Representation: final hidden state's CLS token
- BEiT representation: 768 dimensions
- Selected dimensions: 200, loaded from `final_selected_features.npy`
- Classifier: saved Random Forest, 500 trees
- Class mapping: 0 = Normal, 1 = OSCC

The website is a research/educational prototype and is not a certified clinical diagnostic device.
