# Deploy the API to Modal

Run these commands from the repository root with Python 3.11. The first image build downloads the pinned Cardiff and E5 checkpoints and the fastText language model into the image. Later cold starts use those baked files. The app continues to serve the existing FastAPI routes.

```powershell
python -m pip install -e ".[deploy]"
modal setup
```

Create an ignored `.env.modal` file in the repository root with your values. `OPENROUTER_API_KEY` is required for full analysis. Supply both Upstash variables to share the existing six-hour RSS/discovery cache across containers. `CORS_ORIGINS` is a comma-separated list of exact frontend origins, including `http://localhost:5173` and your actual Vercel production origin. Do not add a trailing slash.

Keep local path settings such as `FASTTEXT_MODEL_PATH` and `APPSTORE_SCAN_DIR` out of this Secret. The Modal entrypoint sets them to the baked model and persistent Volume paths even if an older Secret still contains local values. If the fastText file is unexpectedly absent, the application downloads it on first use.

```dotenv
OPENROUTER_API_KEY=your-key
UPSTASH_REDIS_REST_URL=your-upstash-rest-url
UPSTASH_REDIS_REST_TOKEN=your-upstash-token
CORS_ORIGINS=http://localhost:5173,https://your-actual-vercel-origin
```

Create or update the Modal Secret, then serve a temporary remote development endpoint:

```powershell
modal secret create appstore-reviews-secrets --from-dotenv .env.modal --force
modal serve modal_app.py
```

Use the FastAPI URL printed by `modal serve` to check `/health` and `/docs`. `modal serve` stops when the command exits. For the persistent deployment:

```powershell
modal deploy modal_app.py
modal app dashboard appstore-reviews-analysis
modal app logs appstore-reviews-analysis --tail 100
modal volume ls appstore-reviews-data /
```

`modal deploy` prints the public FastAPI URL; it is also shown in the app dashboard. Verify it with:

```powershell
$apiUrl = "https://<URL-from-modal-deploy>"
Invoke-RestMethod "$apiUrl/health"
Start-Process "$apiUrl/docs"
```

An empty Volume has no `/scans` directory yet. After the first successful `POST /api/scans`, inspect saved scan folders with `modal volume ls appstore-reviews-data /scans`.

Set the Vercel frontend's public `VITE_API_URL` environment variable to that exact API origin and redeploy the frontend. Add the actual Vercel origin to `CORS_ORIGINS` in `.env.modal`, update the Secret with the command above, then redeploy Modal to apply it.

The named `appstore-reviews-data` Volume is created automatically if absent and mounted at `/data`; scan artifacts live under `/data/scans`. The web function commits each scan and queued state before dispatch. The worker reloads that state, commits `running`, then commits completed results or a failure. API reads reload the Volume before accessing scan artifacts. The web function uses one container and one input at a time, and the worker uses one container, avoiding competing writes to `scan.json` in this low-traffic setup. Local Uvicorn still uses ordinary files under `APPSTORE_SCAN_DIR` and FastAPI background tasks.
