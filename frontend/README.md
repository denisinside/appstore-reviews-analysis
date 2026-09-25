# Fieldnotes frontend

React + TypeScript frontend for the App Store Reviews Analysis API. Requires Node.js 20.19+ (or 22.12+) and the repository's Python 3.11+ backend.

## Local setup

In the repository root, start the API:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e ".[sentiment,keywords]"
uvicorn appstore_reviews.api:app --reload --port 8000 --env-file .env
```

Create a root `.env` with `OPENROUTER_API_KEY` before using the command above. The full analysis also needs the fastText language model at `models/lid.176.ftz` (or `FASTTEXT_MODEL_PATH`). See the root README and `ISSUE_ANALYSIS.md` for model setup and analysis requirements. `POST /api/scans` collects reviews before the background analysis begins, so the form can take a while for top-country scans.

In another terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://localhost:5173`. `VITE_API_URL` points to the API origin and defaults to `http://localhost:8000` during local development. Set `VITE_API_URL` to the Modal endpoint in the Vercel project's production environment before building; an unset production value uses the frontend's own origin. The backend's `CORS_ORIGINS` must include the frontend origin (the backend default includes `http://localhost:5173`).

```powershell
npm test
npm run build
```

The production bundle is written to `frontend/dist`. Configure the host to serve `index.html` for both `/` and `/scans/:scanId` so refresh works on the scan route.
