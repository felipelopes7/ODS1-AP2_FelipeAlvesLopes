import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Variáveis globais para armazenar a matriz e o vetorizador após a inicialização
tfidf_matrix = None
item_indices_map = {}

def prepare_data_and_vectorize(items_df: pd.DataFrame):
    """
    Preparação dos Dados e Vetorização.
    Cria a matriz TF-IDF baseada no conteúdo e preenche as globais.
    """
    global tfidf_matrix, item_indices_map
    
    # Limpeza de dados
    items_df['category'] = items_df['category'].fillna('')
    items_df['author'] = items_df['author'].fillna('')
    items_df['title'] = items_df['title'].fillna('')
    items_df['year'] = items_df['year'].astype(str).fillna('')

    # Criando a "Sopa de Metadados"
    items_df['metadata_soup'] = (
        items_df['category'] + " " + 
        items_df['author'] + " " + 
        items_df['year'] + " " + 
        items_df['title'] + " " +
        items_df['tags']
    )

    # Vetorização
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(items_df['metadata_soup'])
    
    # Cria um mapa para achar rapidamente o índice da matriz dado um item_id
    # Ex: item_id 10 -> índice 9 na matriz
    item_indices_map = pd.Series(items_df.index, index=items_df['item_id']).to_dict()
    
    return tfidf_matrix

def build_user_profile(user_id: int, ratings_df: pd.DataFrame) -> np.ndarray:
    """
    Constrói o perfil do usuário (Vetor Médio).
    Pega os itens que o usuário gostou (nota >= 4) e tira a média dos vetores deles.
    """
    # Filtra itens que o usuário gostou
    user_likes = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]
    
    if user_likes.empty:
        return None

    # Pega os índices desses itens na matriz TF-IDF
    liked_indices = []
    for item_id in user_likes['item_id']:
        if item_id in item_indices_map:
            liked_indices.append(item_indices_map[item_id])
            
    if not liked_indices:
        return None

    # Calcula a média dos vetores dos itens curtidos
    # Axis=0 faz a média "vertical" (coluna por coluna), gerando um único vetor resultante
    user_profile = np.mean(tfidf_matrix[liked_indices], axis=0)
    
    # O resultado do np.mean numa matriz esparsa pode vir como matriz 1xN, convertemos para array
    return np.asarray(user_profile)

def get_recommendations(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> list:
    """
    Gera recomendações baseadas na similaridade entre o Perfil do Usuário e os Itens.
    """
    # Garante que a matriz existe (caso a função seja chamada antes da inicialização)
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    # 1. Constrói o perfil
    user_profile = build_user_profile(user_id, ratings_df)
    
    if user_profile is None:
        return [] # Usuário não tem curtidas suficientes para criar perfil

    # 2. Calcula similaridade (Perfil vs Todos os Itens)
    # cosine_similarity aceita (1, n_features) e (n_samples, n_features)
    cosine_sim = cosine_similarity(user_profile, tfidf_matrix)
    
    # O resultado é uma lista de scores [[0.1, 0.5, 0.9...]]
    sim_scores = list(enumerate(cosine_sim[0]))
    
    # 3. Ordena por similaridade (maior para menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 4. Filtra itens já avaliados
    watched_items = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'])
    
    recommendations = []
    for idx, score in sim_scores:
        # Recupera o ID real do item baseando-se no índice do DataFrame
        real_item_id = items_df.iloc[idx]['item_id']
        
        if real_item_id not in watched_items:
            row = items_df.iloc[idx]
            recommendations.append({
                "item_id": int(real_item_id),
                "title": row['title'],
                "category": row['category'],
                "score": float(score)
            })
            
        if len(recommendations) >= 5: # Top 5
            break
            
    return recommendations

def evaluate_accuracy(user_id: int, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Avalia o modelo calculando Precision, Recall e F1-Score.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)

    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    # Precisamos de pelo menos algumas avaliações positivas para testar
    if len(user_ratings[user_ratings['rating'] >= 4]) < 2:
        return {"user_id": user_id, "message": "Poucas avaliações positivas para teste."}

    # Divisão Treino/Teste
    test_items = user_ratings.sample(frac=0.3, random_state=42)
    train_items = user_ratings.drop(test_items.index)
    
    # Recria o dataframe de treino (tudo exceto o teste desse usuário)
    # Nota: Na filtragem baseada em conteúdo, o perfil depende SÓ das avaliações DESTE usuário no treino
    train_df = pd.concat([ratings_df[ratings_df['user_id'] != user_id], train_items])

    # Gera recomendações baseadas APENAS no treino
    recs = get_recommendations(user_id, items_df, train_df)
    
    recommended_ids = {r['item_id'] for r in recs}
    
    # Ground Truth: Itens do conjunto de teste que o usuário GOSTOU (nota >= 4)
    relevant_items = set(test_items[test_items['rating'] >= 4]['item_id'])
    
    if not relevant_items:
         return {"user_id": user_id, "message": "Não há itens relevantes no conjunto de teste."}

    # Interseção (Acertos)
    hits = len(recommended_ids & relevant_items)
    
    # --- CÁLCULO DAS MÉTRICAS (Exigido pelo PDF) ---
    
    # Precision: Dos itens recomendados, quantos eram relevantes?
    # (Evita divisão por zero se não recomendou nada)
    precision = hits / len(recommended_ids) if len(recommended_ids) > 0 else 0.0
    
    # Recall: Dos itens relevantes existentes, quantos foram recomendados?
    recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    # F1-Score: Média harmônica
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    return {
        "user_id": user_id,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "hits": hits,
        "recommended": list(recommended_ids),
        "relevant_in_test": list(relevant_items)
    }

def calculate_overall_accuracy(items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> dict:
    """
    Média das métricas para todos os usuários.
    """
    if tfidf_matrix is None:
        prepare_data_and_vectorize(items_df)
        
    unique_users = ratings_df["user_id"].unique()
    metrics = {"precision": [], "recall": [], "f1_score": []}

    for user_id in unique_users:
        result = evaluate_accuracy(user_id, items_df, ratings_df)
        if "f1_score" in result:
            metrics["precision"].append(result["precision"])
            metrics["recall"].append(result["recall"])
            metrics["f1_score"].append(result["f1_score"])

    if not metrics["f1_score"]:
        return {"message": "Não foi possível calcular métricas para nenhum usuário."}

    return {
        "mean_precision": sum(metrics["precision"]) / len(metrics["precision"]),
        "mean_recall": sum(metrics["recall"]) / len(metrics["recall"]),
        "mean_f1_score": sum(metrics["f1_score"]) / len(metrics["f1_score"]),
        "users_evaluated": len(metrics["f1_score"])
    }