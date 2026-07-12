# Deploy the live resume link

This project is configured as **one Render web service**.  The same URL serves
the dashboard UI and the FastAPI endpoints, so the buttons use the real API
rather than local Streamlit logic.

1. Push this repository to GitHub.
2. In [Render](https://render.com), select **New +** > **Blueprint** and choose
   this repository. Render detects `render.yaml` automatically.
3. Click **Apply**. After the build succeeds, copy the service URL shown by
   Render (for example, `https://rainfall-crop-prediction.onrender.com`).
4. Put that HTTPS URL in your resume. API proof is available at
   `https://YOUR-URL/docs` and a health check at `https://YOUR-URL/health`.

For a custom URL, change the `name` value in `render.yaml` before creating the
service, if that name is available. The free Render tier can sleep after
inactivity; the first visit may take a short time to wake up.

## Local check

```powershell
python -m uvicorn nawsp.api.main:app --reload
```

Open `http://127.0.0.1:8000`. The dashboard calls FastAPI at the same address.
