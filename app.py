"""
Production FastAPI app: Predict + Explain endpoints for Cricket Win Predictions.

- POST /predict
    multipart/form-data
    file: CSV upload
    returns JSON with saved predictions file path and metadata

- POST /explain/{prediction_id}
    Path param prediction_id:
      - exact filename (present in data/api_results), OR
      - integer (index) meaning: 'use latest results file and pick row by index'
    Optional JSON body: {"row_index": <int>} to pick a specific row from a specified file.
    Returns: prediction, confidence, explanation, and some metadata.

Requirements:
- Place trained artifacts (best_model.pkl, scaler.pkl, model_metadata.json) under models/current/
  OR keep them at project root (fallback).
- Add OPENAI_API_KEY to .env
- Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, Path as FPath, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------
# Load env, Configure OpenAI
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # We won't throw immediately; endpoints that call OpenAI will raise a clear error.
    pass
else:
    openai.api_key = OPENAI_API_KEY

# ----------------------------
# Basic configuration & logging
# ----------------------------
APP_NAME = "neural_lines_cricket_api"
BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "current"
FALLBACK_MODEL_DIR = BASE_DIR  # fallback if user saved artifacts at project root
RESULTS_DIR = BASE_DIR / "data" / "api_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"{APP_NAME}.log"

logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# avoid duplicate handlers in dev
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Cricket Win Prediction API with Explanations", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Model bundle helper
# ----------------------------
class ModelBundle(BaseModel):
    model_name: str
    model_version: str
    feature_names: List[str]
    model_path: str
    scaler_path: str

def _load_bundle_from_dir(model_dir: Path) -> Optional[ModelBundle]:
    model_path = model_dir / "best_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "model_metadata.json"
    if not (model_path.exists() and scaler_path.exists() and metadata_path.exists()):
        return None
    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        model_name = meta.get("best_model_name", "unknown_model")
        feature_names = meta.get("feature_names") or []
        model_version = meta.get("model_version") or model_dir.name
        return ModelBundle(
            model_name=model_name,
            model_version=model_version,
            feature_names=feature_names,
            model_path=str(model_path),
            scaler_path=str(scaler_path),
        )
    except Exception as e:
        logger.exception(f"Failed reading metadata from {metadata_path}: {e}")
        return None

def resolve_model_bundle(requested: Optional[str] = None) -> ModelBundle:
    if requested:
        candidate = BASE_DIR / "models" / requested
        bundle = _load_bundle_from_dir(candidate)
        if bundle:
            return bundle
        raise HTTPException(status_code=400, detail=f"Requested model '{requested}' not found or incomplete.")
    bundle = _load_bundle_from_dir(DEFAULT_MODEL_DIR)
    if bundle:
        return bundle
    bundle = _load_bundle_from_dir(FALLBACK_MODEL_DIR)
    if bundle:
        return bundle
    raise HTTPException(status_code=500, detail="No valid model artifacts found. Ensure best_model.pkl, scaler.pkl, and model_metadata.json exist.")

# ----------------------------
# Feature engineering (mirror training)
# ----------------------------
REQUIRED_COLUMNS = ["total_runs", "wickets", "target", "balls_left"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balls_left"] = df["balls_left"].clip(lower=0)
    df["required_runs"] = df["target"] - df["total_runs"]
    df["wickets_remaining"] = 10 - df["wickets"]

    overs_left = df["balls_left"] / 6
    overs_bowled = (120 - df["balls_left"]) / 6
    overs_bowled = overs_bowled.replace(0, 1e-5)

    df["required_rr"] = np.where(
        df["balls_left"] > 0,
        df["required_runs"] / overs_left,
        np.where(df["required_runs"] > 0, np.inf, 0)
    )
    df["current_rr"] = df["total_runs"] / overs_bowled

    df["pressure_index"] = (df["required_rr"] * (11 - df["wickets_remaining"])) / 10
    df["run_rate_difference"] = df["required_rr"] - df["current_rr"]
    df["balls_per_wicket_remaining"] = np.where(
        df["wickets_remaining"] > 0, df["balls_left"] / df["wickets_remaining"], 0
    )

    max_realistic_rr = 12
    df["theoretical_max_runs"] = df["balls_left"] * (max_realistic_rr / 6)
    # protect divide by zero
    df["win_feasibility"] = np.where(
        df["theoretical_max_runs"] > 0,
        np.where(df["required_runs"] <= df["theoretical_max_runs"],
                 1 - (df["required_runs"] / df["theoretical_max_runs"]),
                 0),
        0
    )
    df["win_feasibility"] = df["win_feasibility"].fillna(0)

    overs_completed = (120 - df["balls_left"]) / 6
    df["match_phase"] = np.where(overs_completed <= 6, 0, np.where(overs_completed <= 15, 1, 2))

    df = df.replace([np.inf, -np.inf], np.nan)
    df["required_rr"] = df["required_rr"].fillna(0).clip(0, 36)
    df["pressure_index"] = df["pressure_index"].fillna(0).clip(0, 100)

    return df

# ----------------------------
# Response schemas
# ----------------------------
class PredictResponse(BaseModel):
    status: str
    predictions_file: Optional[str]
    metadata: dict

class ExplainRequest(BaseModel):
    # optional override of row_index when path param is a filename
    row_index: Optional[int] = None
    # optional small instruction to tailor explanation (not required)
    style: Optional[str] = None

class ExplainResponse(BaseModel):
    prediction: int
    confidence: float
    explanation: str
    source_file: str
    row_index: int

# ----------------------------
# /predict endpoint
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), model: Optional[str] = None):
    request_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    logger.info(f"/predict called | filename={file.filename} | content_type={file.content_type} | request_id={request_id}")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    bundle = resolve_model_bundle(requested=model)
    logger.info(f"Using model bundle: name={bundle.model_name} version={bundle.model_version}")

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logger.exception(f"Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail="Malformed CSV. Unable to parse.")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}. Required={REQUIRED_COLUMNS}")

    total_rows = len(df)
    filtered = df.query("balls_left < 60 and target > 120").copy()
    filtered_rows = len(filtered)

    if filtered_rows == 0:
        meta = {
            "total_rows": total_rows,
            "filtered_rows": 0,
            "predictions_made": 0,
            "model_used": f"{bundle.model_name}:{bundle.model_version}",
        }
        logger.info(f"No rows matched filter. {meta}")
        return PredictResponse(status="success", predictions_file=None, metadata=meta)

    fe = engineer_features(filtered)

    try:
        scaler = joblib.load(bundle.scaler_path)
        model_obj = joblib.load(bundle.model_path)
    except Exception as e:
        logger.exception(f"Failed loading model artifacts: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model artifacts.")

    feature_names = bundle.feature_names or [
        "total_runs", "wickets", "target", "balls_left",
        "required_runs", "wickets_remaining", "required_rr", "current_rr",
        "pressure_index", "run_rate_difference", "balls_per_wicket_remaining",
        "win_feasibility", "match_phase"
    ]

    missing_feats = [c for c in feature_names if c not in fe.columns]
    if missing_feats:
        logger.error(f"Engineered features missing: {missing_feats}")
        raise HTTPException(status_code=500, detail=f"Engineered features missing: {missing_feats}. Check FE pipeline.")

    X = fe[feature_names]
    try:
        X_scaled = scaler.transform(X)
        proba = model_obj.predict_proba(X_scaled)[:, 1]
        preds = (proba >= 0.5).astype(int)
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")

    out = filtered.copy()
    out["predicted_won"] = preds
    out["win_probability"] = proba

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"predictions_{ts}_{request_id}.csv"
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        logger.exception(f"Failed writing results: {e}")
        raise HTTPException(status_code=500, detail="Failed to write results CSV.")

    meta = {
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "predictions_made": int(out.shape[0]),
        "model_used": f"{bundle.model_name}:{bundle.model_version}",
    }

    logger.info(f"Success | results={out_path} | meta={meta}")
    return PredictResponse(status="success", predictions_file=str(out_path.resolve()), metadata=meta)

# ----------------------------
# Helper: locate results file & row
# ----------------------------
def _list_results_files() -> List[Path]:
    files = sorted(RESULTS_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def _resolve_prediction_file_and_row(prediction_id: str, row_index_override: Optional[int] = None) -> (Path, int):
    """
    prediction_id:
      - if it's a filename that exists in RESULTS_DIR -> use it
      - elif it is numeric string -> interpret as row_index on the latest file
      - else raise
    row_index_override: if provided and filename selected, use it instead
    returns: (file_path, row_index)
    """
    # if filename matches exactly
    candidate = RESULTS_DIR / prediction_id
    if candidate.exists() and candidate.is_file():
        chosen_file = candidate
        if row_index_override is not None:
            ri = row_index_override
        else:
            ri = 0  # default to first row if none provided
        return chosen_file, ri

    # try numeric index for the latest file
    try:
        idx = int(prediction_id)
        files = _list_results_files()
        if not files:
            raise HTTPException(status_code=404, detail="No prediction result files available on server.")
        chosen_file = files[0]  # most recent file
        if idx < 0 or idx >= sum(1 for _ in pd.read_csv(chosen_file).itertuples()):
            raise HTTPException(status_code=400, detail=f"Requested row index {idx} is out of range for latest file {chosen_file.name}.")
        return chosen_file, idx
    except ValueError:
        # not numeric - maybe they forgot extension, try fuzzy match
        files = _list_results_files()
        matches = [f for f in files if prediction_id in f.name]
        if len(matches) == 1:
            chosen_file = matches[0]
            ri = row_index_override if row_index_override is not None else 0
            return chosen_file, ri
        elif len(matches) > 1:
            raise HTTPException(status_code=400, detail=f"Ambiguous prediction_id; multiple files match: {[p.name for p in matches]}. Use exact filename or numeric index.")
        else:
            raise HTTPException(status_code=404, detail=f"Prediction file not found: {prediction_id}. Available: {[p.name for p in files]}")

# ----------------------------
# Prompting & OpenAI wrapper
# ----------------------------
def build_explanation_prompt(row: pd.Series, prediction: int, confidence: float, style: Optional[str] = None) -> str:
    """
    Build a compact, clear prompt describing the match state and model output.
    The prompt instructs the model to produce 2-4 sentence human-readable explanation.
    """
    # Select fields to include (avoid flooding)
    fields = {
        "target": int(row.get("target", np.nan)),
        "runs_scored": float(row.get("total_runs", np.nan)),
        "balls_left": int(row.get("balls_left", np.nan)),
        "wickets_out": float(row.get("wickets", np.nan)),
        "wickets_remaining": float(row.get("wickets_remaining", np.nan)) if "wickets_remaining" in row.index else None,
        "required_runs": float(row.get("required_runs", np.nan)) if "required_runs" in row.index else None,
        "required_rr": float(row.get("required_rr", np.nan)) if "required_rr" in row.index else None,
        "current_rr": float(row.get("current_rr", np.nan)) if "current_rr" in row.index else None,
        "win_feasibility": float(row.get("win_feasibility", np.nan)) if "win_feasibility" in row.index else None,
    }

    # Base instruction
    prompt = [
        "You are an expert cricket analyst and communicator. Given a match situation and a model's prediction, produce a concise (2-4 sentence) human-readable explanation for a non-technical audience.",
        "",
        "MATCH SITUATION:"
    ]
    for k, v in fields.items():
        if v is None or (isinstance(v, float) and (pd.isna(v) or np.isinf(v))):
            continue
        prompt.append(f"- {k.replace('_', ' ').title()}: {v}")

    prompt.append("")
    # Model output
    pred_text = "win" if prediction == 1 else "lose"
    prompt.append(f"MODEL OUTPUT: predicted outcome = {pred_text}; probability = {confidence:.2f}")
    prompt.append("")
    # Conditional guidance based on confidence
    if confidence >= 0.7:
        prompt.append("GUIDANCE: This is a high-confidence prediction. Explain clearly why the batting team is (likely/unlikely) to succeed, referencing runs required, balls left, and wickets.")
    elif confidence <= 0.3:
        prompt.append("GUIDANCE: Low-confidence prediction. Explain the main reasons the batting team is unlikely to win and highlight key constraints (e.g., few balls left, many runs required, few wickets remaining).")
    else:
        prompt.append("GUIDANCE: Moderate confidence. Explain that the match is close, outline the key uncertainties, and list the main factors that could swing the result (e.g., a quick boundary, a wicket).")

    if style:
        prompt.append(f"STYLE: {style.strip()}")

    prompt.append("")
    prompt.append("Avoid jargon. Keep it simple and actionable. Do not invent facts (e.g., do not mention specific players or conditions that are not in the data).")

    return "\n".join(prompt)

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
def call_openai_for_explanation(prompt: str, max_tokens: int = 150, temperature: float = 0.2) -> str:
    """
    Calls OpenAI's chat completion API and returns a string reply.
    Requires OPENAI_API_KEY to be set.
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured (OPENAI_API_KEY missing).")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured on the server.")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o if available
            messages=[
                {"role": "system", "content": "You are a concise cricket insights assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API request failed: {str(e)}")


# ----------------------------
# /explain endpoint
# ----------------------------
PREDICTIONS_FILE = "/Users/SHAH/Work/neural_lines_assement/data/predictions.csv"

@app.post("/explain/{row_index}")
def explain_prediction(row_index: int):
    try:
        df = pd.read_csv(PREDICTIONS_FILE)
    except Exception as e:
        logger.exception(f"Could not read predictions file {PREDICTIONS_FILE}")
        raise HTTPException(status_code=500, detail=f"Failed to read predictions file: {str(e)}")

    if row_index < 0 or row_index >= len(df):
        raise HTTPException(status_code=404, detail=f"Row index {row_index} out of range. File has {len(df)} rows.")

    row = df.iloc[row_index]
    prediction = int(row.get("prediction", -1))
    confidence = float(row.get("confidence", 0.0))
    match_context = row.to_dict()

    # Build prompt for LLM
    prompt = f"""
    Match context: {match_context}
    Model prediction: {prediction}
    Win probability: {confidence:.2f}
    Provide a human-readable explanation (2-3 sentences).
    """
    explanation = call_openai_for_explanation(prompt)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "explanation": explanation,
        "source_file": PREDICTIONS_FILE,
        "row_index": row_index
    }


# ----------------------------
# Health & model endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/model")
def model_info(model: Optional[str] = None):
    bundle = resolve_model_bundle(requested=model)
    return {
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "model_path": bundle.model_path,
        "scaler_path": bundle.scaler_path,
        "feature_names": bundle.feature_names,
    }

# ----------------------------
# If run as script
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
