from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from shapash.utils.load_smartpredictor import load_smartpredictor

BASE_DIR = Path(__file__).resolve().parent
PREDICTOR_PATH = BASE_DIR / 'predictor_fastapi.pkl'
predictor_load = load_smartpredictor(str(PREDICTOR_PATH))


class TitanicPassenger(BaseModel):
    Pclass: int
    Age: float
    Sex: str
    SibSp: int
    Parch: int


def cast_input_to_predictor_schema(x_input: pd.DataFrame) -> pd.DataFrame:
    expected_types = predictor_load.features_types.copy()
    x_cast = x_input[list(expected_types.keys())].copy()

    for col, dtype in expected_types.items():
        if dtype.startswith('int') or dtype.startswith('float'):
            x_cast[col] = pd.to_numeric(x_cast[col], errors='coerce').astype(dtype)
        else:
            x_cast[col] = x_cast[col].astype(dtype)
    return x_cast


app = FastAPI(title='Shapash SmartPredictor API')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: TitanicPassenger):
    x_input = pd.DataFrame([payload.model_dump()])
    x_input = cast_input_to_predictor_schema(x_input)
    predictor_load.add_input(x=x_input)

    ypred_df = predictor_load.data['ypred']
    ypred_value = ypred_df.iloc[0, 0]

    proba_df = predictor_load.predict_proba()
    proba_dict = proba_df.iloc[0].to_dict()

    return {
        'prediction': ypred_value,
        'predict_proba': proba_dict,
    }


@app.post('/explain')
def explain(payload: TitanicPassenger):
    x_input = pd.DataFrame([payload.model_dump()])
    x_input = cast_input_to_predictor_schema(x_input)
    predictor_load.add_input(x=x_input)
    predictor_load.modify_mask(max_contrib=4)

    detail = predictor_load.detail_contributions()
    summary = predictor_load.summarize()

    if hasattr(detail, 'to_dict'):
        detail_out = detail.to_dict(orient='records')[0]
    else:
        detail_out = detail

    if hasattr(summary, 'to_dict'):
        summary_out = summary.to_dict(orient='records')[0]
    else:
        summary_out = summary

    return {
        'detail_contributions': detail_out,
        'summary': summary_out,
    }
