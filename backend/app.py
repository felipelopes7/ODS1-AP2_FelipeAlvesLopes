from fastapi import FastAPI
import pandas as pd
from recommender import get_recommendations, evaluate_accuracy, calculate_overall_accuracy

app = FastAPI()

items_df = pd.read_csv("items.csv")

@app.get("/")
def root():
    return {"message": "Manga Recommender API online"}

@app.get("/recomendar/{user_id}")
def recomendar(user_id: int):
    # Recarrega o arquivo de avaliações a cada chamada
    ratings_df = pd.read_csv("ratings.csv")
    recs = get_recommendations(user_id, items_df, ratings_df)
    return {"user_id": user_id, "recommendations": recs}

@app.get("/avaliar_acuracia/{user_id}")
def avaliar_acuracia(user_id: int):
    # Recarrega o arquivo de avaliações a cada chamada
    ratings_df = pd.read_csv("ratings.csv")
    result = evaluate_accuracy(user_id, items_df, ratings_df)
    if "message" in result:
        return {"message": result["message"]}
    return result

@app.get("/avaliar_acuracia_geral")
def avaliar_acuracia_geral():
    # Recarrega o arquivo de avaliações a cada chamada
    ratings_df = pd.read_csv("ratings.csv")
    result = calculate_overall_accuracy(items_df, ratings_df)
    return result