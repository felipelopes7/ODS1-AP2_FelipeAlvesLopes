import pandas as pd
import numpy as np

def cosine_similarity_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Calcula a similaridade cosseno entre as linhas da matriz mat.
    Retorna uma matriz de similaridade quadrada.
    """
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1e-9  # evita divisão por zero
    normalized = mat / norms[:, None]
    return normalized @ normalized.T  # produto escalar para similaridade

def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a matriz usuário-item a partir do dataframe de avaliações.
    Valores ausentes são preenchidos com 0.
    """
    # Garantir que IDs e ratings sejam inteiros
    ratings_df["user_id"] = pd.to_numeric(ratings_df["user_id"], errors="coerce").fillna(0).astype(int)
    ratings_df["item_id"] = pd.to_numeric(ratings_df["item_id"], errors="coerce").fillna(0).astype(int)
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce").fillna(0).astype(int)

    return ratings_df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

def get_recommendations(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> list:
    """
    Gera 5 recomendações de itens para um usuário usando filtragem colaborativa baseada em itens.
    """
    ui_matrix = build_user_item_matrix(ratings_df)

    if user_id not in ui_matrix.index:
        return []  # usuário não possui avaliações

    item_matrix = ui_matrix.T.values
    sim_matrix = cosine_similarity_matrix(item_matrix)
    item_sim = pd.DataFrame(sim_matrix, index=ui_matrix.columns, columns=ui_matrix.columns)

    preds = {}
    for item in ui_matrix.columns:
        if ui_matrix.loc[user_id, item] != 0:
            continue  # usuário já avaliou este item

        sims = item_sim[item]
        rated_mask = ui_matrix.loc[user_id] != 0
        weights = sims[rated_mask.index[rated_mask]]
        ratings = ui_matrix.loc[user_id, rated_mask]

        if len(weights) == 0:
            continue  # nenhum item similar avaliado

        score = (weights * ratings).sum() / (np.abs(weights).sum() + 1e-9)
        preds[item] = score

    # Seleciona os 5 top itens
    top_n = 5
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for item_id, score in top_items:
        row = items_df[items_df['item_id'] == item_id]
        if not row.empty:
            results.append({
                "item_id": int(item_id),
                "title": row['title'].values[0],
                "category": row['category'].values[0],
                "score": float(score)
            })

    return results

def evaluate_accuracy(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Avalia a acurácia da recomendação dividindo as avaliações do usuário em treino e teste (70/30).
    Acurácia = (número de acertos) / (número de itens recomendados).
    """
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]

    if len(user_ratings) < 2:
        return {"user_id": user_id, "message": "Usuário não tem avaliações suficientes para o teste de acurácia (requer pelo menos 2 avaliações)."}

    # Divisão treino/teste (70% treino, 30% teste)
    test_fraction = 0.3
    test_size = max(1, int(len(user_ratings) * test_fraction))
    test_items = user_ratings.sample(test_size, random_state=42)
    train_items = user_ratings.drop(test_items.index)

    # Dataframe de treino contendo todas as avaliações de outros usuários + treino do usuário atual
    train_df = pd.concat([ratings_df[ratings_df['user_id'] != user_id], train_items])

    # Gera 5 recomendações com base no treino
    recs = get_recommendations(user_id, items_df, train_df)

    if not recs:
        return {"user_id": user_id, "message": "Nenhuma recomendação encontrada para o usuário com base nos dados de treino."}
    
    recommended_ids = {r['item_id'] for r in recs}
    test_liked = set(test_items[test_items['rating'] >= 4]['item_id'])  # >=4 considera "gostou"

    if not test_liked:
        return {"user_id": user_id, "message": "Nenhuma avaliação positiva (>=4) encontrada no conjunto de teste. A acurácia não pode ser calculada."}

    hits = len(recommended_ids & test_liked)
    
    # === REVERSÃO DA LÓGICA: Denominador é SEMPRE o número de itens recomendados (5) ===
    denominator = len(recs)
    accuracy = hits / denominator # Precisão: hits / total de recomendações
    # =================================================================================

    return {
        "user_id": user_id,
        "recommended": list(recommended_ids),
        "test_liked": list(test_liked),
        "hits": hits,
        "accuracy": accuracy
    }

def calculate_overall_accuracy(items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Calcula a acurácia média do modelo para todos os usuários usando a métrica Precision.
    """
    unique_users = ratings_df["user_id"].unique()
    all_accuracies = []

    for user_id in unique_users:
        result = evaluate_accuracy(user_id, items_df, ratings_df)
        if "accuracy" in result:
            all_accuracies.append(result["accuracy"])

    if not all_accuracies:
        return {"message": "Nenhum usuário com dados suficientes para calcular a acurácia."}

    overall_accuracy = sum(all_accuracies) / len(all_accuracies)
    return {
        "overall_accuracy": overall_accuracy,
        "total_users_evaluated": len(all_accuracies)
    }
