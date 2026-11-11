#from fastapi import FastAPI
#from pydantic import BaseModel
#from database import init_db, save_log

#### init_db()

#class PricingRequest(BaseModel):
#    product_id: str
 ####@app.post("/predict")
#def predict_price_effect(data: PricingRequest):
 #   price_change = data.proposed_price - data.current_price
    
    # Simple multiplier placeholder (baad me model connect karenge)
  #  estimated_effect = float(-1.35 * price_change) 

    # SQLite Database me entry save karna
   # save_log(
    #    product_id=data.product_id,
    #    current_price=data.current_price,
    #    proposed_price=data.proposed_price,
    #    effect=estimated_effect
    #)

   # return {
   #     "status": "Saved to Database",
   #     "product_id": data.product_id,
    #    "estimated_causal_effect": round(estimated_effect, 4)
    #}
import sys
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import init_db, save_log

app = FastAPI(title="EconoCausal Backend API")

# Global reference for trained DML causal effect results
dml_mean_effect = -57.18  # Calculated average price elasticity / CATE from DML training

@app.on_event("startup")
def setup_database_and_model():
    init_db()
    print("Database initialized successfully.")

class PricingRequest(BaseModel):
    product_id: str
    current_price: float
    proposed_price: float
    features: list[float]  # Confounders (X) e.g., market competition, seasonality

@app.post("/predict")
def predict_price_effect(data: PricingRequest):
    try:
        # Price change calculation
        price_change = data.proposed_price - data.current_price
        
        # Calculate causal demand impact using trained DML effect coefficient
        estimated_effect = float(dml_mean_effect * price_change)

        # Save record into SQLite database (econocausal.db)
        save_log(
            product_id=data.product_id,
            current_price=data.current_price,
            proposed_price=data.proposed_price,
            effect=estimated_effect
        )

        return {
            "status": "Saved to Database",
            "product_id": data.product_id,
            "price_change": price_change,
            "estimated_causal_effect": round(estimated_effect, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
