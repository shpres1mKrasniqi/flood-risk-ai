# Flood risk frontend

React + Vite + Tailwind interface for the flood-risk API. Albanian by default, with English.

## Run

1. Start the backend (from `backend/`): `uv run fastapi dev app/main.py`
2. Start the frontend (from `frontend/`):
   ```bash
   npm install
   npm run dev
   ```
3. Open http://localhost:5173

The backend URL defaults to `http://127.0.0.1:8000`. To change it, copy `.env.example` to `.env` and set `VITE_API_URL`.

## Structure

- `src/features/flood-risk/services/floodRiskApi.js` - calls `/api/flood-risk/metadata` and `/api/flood-risk/explain`
- `src/features/flood-risk/components/` - form, result, staff gauge, explanation, limitations
- `src/features/flood-risk/utils/fields.js` - form fields, Malisheva defaults, validation
- `src/i18n/translations.js` - all interface text (sq, en)
- `src/store/languageStore.js` - selected language (remembered in the browser)
