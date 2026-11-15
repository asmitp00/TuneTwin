# TuneTwin web app image (used by the Hugging Face Docker Space).
# Python 3.12 + scikit-learn 1.9.0 match the environment that trained
# models/tunetwin_recommender.joblib.
FROM python:3.12-slim

# Hugging Face runs Space containers as user ID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# Only what the web app reads at runtime (no notebooks, zip or test artifacts).
COPY --chown=user app.py storage.py ./
COPY --chown=user templates templates
COPY --chown=user static static
COPY --chown=user models/tunetwin_recommender.joblib models/
COPY --chown=user data/data.csv data/
COPY --chown=user data/extracted/data/data_w_genres.csv data/extracted/data/
RUN mkdir -p instance

EXPOSE 7860
# One worker keeps a single in-memory copy of the catalog; threads serve
# concurrent listeners. The timeout covers the ~15 s catalog load at boot.
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 8 --timeout 120 --bind 0.0.0.0:${PORT}"]
