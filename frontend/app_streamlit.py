import streamlit as st
import pandas as pd
import os
import requests
import altair as alt
import math
from streamlit_extras.card import card
from streamlit_option_menu import option_menu

# --- Configuração da Página ---
st.set_page_config(
    page_title="Recomendador de Mangás",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Exibição de Notificações Agendadas ---
if 'toast_message' in st.session_state and st.session_state.toast_message:
    st.toast(st.session_state.toast_message['message'], icon=st.session_state.toast_message['icon'])
    # Limpa a mensagem para não exibir novamente
    st.session_state.toast_message = None

# --- Inicializa o estado da sessão ---
if 'selected_manga_id' not in st.session_state:
    st.session_state.selected_manga_id = None
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = 1
if 'page' not in st.session_state:
    st.session_state.page = 1

# --- Constantes e Carregamento de Dados ---
ITEMS_CSV = "../backend/items.csv"
RATINGS_CSV = "../backend/ratings.csv"
API_URL = "http://127.0.0.1:8000"

@st.cache_data
def load_data():
    """Carrega os dados dos arquivos CSV, com cache para performance."""
    items = pd.read_csv(ITEMS_CSV)
    if os.path.exists(RATINGS_CSV):
        ratings = pd.read_csv(RATINGS_CSV)
        for col in ["user_id", "item_id", "rating"]:
            ratings[col] = pd.to_numeric(ratings[col], errors="coerce").fillna(0).astype(int)
    else:
        ratings = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    return items, ratings

def get_items_with_avg(items_df, ratings_df):
    """Calcula a média de avaliação para cada item."""
    if ratings_df.empty:
        items_with_avg = items_df.copy()
        items_with_avg['avg_rating'] = 0
        return items_with_avg
        
    avg_ratings = ratings_df.groupby("item_id")["rating"].mean().reset_index()
    items_with_avg = items_df.merge(avg_ratings, on="item_id", how="left").fillna(0)
    items_with_avg.rename(columns={"rating": "avg_rating"}, inplace=True)
    return items_with_avg

items_df, ratings_df = load_data()
items_with_avg = get_items_with_avg(items_df, ratings_df)

# --- Funções Auxiliares ---
def set_selected_manga_and_rerun(item_id):
    """Define o mangá selecionado e força a re-renderização para a página de detalhes."""
    st.session_state.selected_manga_id = item_id
    st.rerun()

# --- Funções das Páginas ---

def display_catalog():
    st.header(" Catálogo de Mangás")
    # ... (código do catálogo permanece o mesmo)
    search_query = st.text_input("Buscar por título", key="search_input")
    categories = ["Todas"] + sorted(items_with_avg["category"].unique().tolist())
    selected_category = st.selectbox("Filtrar por Categoria", options=categories, key="category_select")

    filtered_items = items_with_avg
    if search_query:
        filtered_items = filtered_items[filtered_items["title"].str.contains(search_query, case=False, na=False)]
    if selected_category != "Todas":
        filtered_items = filtered_items[filtered_items["category"] == selected_category]

    if filtered_items.empty:
        st.warning("Nenhum mangá encontrado.")
    else:
        ITEMS_PER_PAGE = 12
        total_pages = math.ceil(len(filtered_items) / ITEMS_PER_PAGE)
        if st.session_state.page > total_pages and total_pages > 0:
            st.session_state.page = 1
        
        start_idx = (st.session_state.page - 1) * ITEMS_PER_PAGE
        paginated_items = filtered_items.iloc[start_idx : start_idx + ITEMS_PER_PAGE]

        for i in range(0, len(paginated_items), 4):
            cols = st.columns(4)
            row_items = paginated_items.iloc[i:i+4]
            for j, (_, row) in enumerate(row_items.iterrows()):
                with cols[j]:
                    card(
                        title=f"{row['title']}",
                        text=f"⭐ {row['avg_rating']:.2f}" if row['avg_rating'] > 0 else "Sem avaliações",
                        image=row['image_url'],
                        on_click=lambda item_id=row['item_id']: set_selected_manga_and_rerun(item_id),
                        key=f"card_{row['item_id']}",
                        styles={
                            "card": {"width": "100%", "height": "400px", "margin": "0px"},
                            "title": {"line-height": "1.2em"}
                        }
                    )
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        if col1.button("Anterior", disabled=(st.session_state.page <= 1), use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
        col2.markdown(f"<div style='text-align: center; margin-top: 10px;'>Página {st.session_state.page} de {total_pages}</div>", unsafe_allow_html=True)
        if col3.button("Próxima", disabled=(st.session_state.page >= total_pages), use_container_width=True):
            st.session_state.page += 1
            st.rerun()


def display_add_rating():
    """Renderiza a página para adicionar ou atualizar avaliações."""
    st.header(" Adicionar ou Atualizar Avaliação")
    global ratings_df
    
    new_user_id = st.number_input("ID do Usuário", min_value=1, step=1, value=st.session_state.current_user_id)
    manga_titles = items_df["title"].tolist()
    selected_manga_title = st.selectbox("Nome do Mangá", manga_titles)
    new_item_id = items_df[items_df["title"] == selected_manga_title]["item_id"].iloc[0]
    st.write(f"ID do Mangá Selecionado: {new_item_id}")
    new_rating = st.slider("Nota do Mangá", 1, 5, 3)

    if st.button("Salvar Avaliação"):
        exists_index = ratings_df[
            (ratings_df["user_id"] == new_user_id) & (ratings_df["item_id"] == new_item_id)
        ].index

        if not exists_index.empty:
            ratings_df.loc[exists_index, "rating"] = new_rating
            st.session_state.toast_message = {"message": "✅ Avaliação atualizada com sucesso!", "icon": "✅"}
        else:
            new_row = pd.DataFrame([{"user_id": new_user_id, "item_id": new_item_id, "rating": new_rating}])
            ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
            st.session_state.toast_message = {"message": "✅ Avaliação adicionada com sucesso!", "icon": "✅"}

        ratings_df.to_csv(RATINGS_CSV, index=False)
        st.cache_data.clear()

        # Mostra o aviso imediatamente
        st.toast(st.session_state.toast_message["message"], icon=st.session_state.toast_message["icon"])
        st.session_state.toast_message = None

        st.subheader("Avaliações existentes")
        st.dataframe(ratings_df)

def display_recommendations():
    """Renderiza a página de geração de recomendações."""
    st.header(" Gerar Recomendações")
    if ratings_df.empty:
        st.info("Adicione avaliações para gerar recomendações.")
    else:
        user_ids = sorted(ratings_df["user_id"].unique())
        selected_user = st.selectbox("Escolha o ID do Usuário", options=user_ids)
        st.write("**Quantidade de Recomendações:** 5")
        if st.button("Gerar Recomendações"):
            with st.spinner('Buscando recomendações...'):
                try:
                    response = requests.get(f"{API_URL}/recomendar/{selected_user}")
                    response.raise_for_status() 
                    recs = response.json().get("recommendations", [])
                    if recs:
                        rec_df = pd.DataFrame(recs).merge(items_df[['item_id', 'image_url']], on='item_id')
                        st.subheader(f"Recomendações para Usuário {selected_user}")
                        cols = st.columns(len(rec_df))
                        for i, (_, row) in enumerate(rec_df.iterrows()):
                            with cols[i]:
                                st.image(row['image_url'], caption=row['title'], use_container_width=True)
                                st.write(f"**Score:** {row['score']:.2f}")
                    else:
                        st.warning("Nenhuma recomendação encontrada para este usuário.")
                except requests.RequestException as e:
                    st.error(f"Erro de conexão com o backend: {e}")

def display_accuracy():
    """Renderiza a página de avaliação de performance (Precision, Recall, F1)."""
    st.header(" Avaliar Performance")
    
    # Verifica se há avaliações
    if ratings_df.empty or ratings_df['user_id'].nunique() < 1:
        st.info("Adicione mais avaliações para calcular as métricas.")
    else:
        st.subheader("Avaliação por Usuário")
        user_ids = sorted(ratings_df["user_id"].unique())
        selected_user = st.selectbox("Escolha um usuário", options=user_ids)
        st.write("**Parâmetros:** Consideramos 'relevante' notas >= 4.")
        
        if st.button("Calcular Métricas do Usuário"):
            with st.spinner("Calculando..."):
                try:
                    response = requests.get(f"{API_URL}/avaliar_acuracia/{selected_user}")
                    response.raise_for_status()
                    result = response.json()
                    
                    if "message" in result:
                        st.warning(result["message"])
                    else:
                        # --- MUDANÇA AQUI: Exibe as 3 métricas ---
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Precision", f"{result.get('precision', 0):.2%}", help="Dos recomendados, quantos o usuário curtiu?")
                        col2.metric("Recall", f"{result.get('recall', 0):.2%}", help="Dos que o usuário curte, quantos encontramos?")
                        col3.metric("F1-Score", f"{result.get('f1_score', 0):.2%}", help="Média harmônica")
                        
                        st.markdown("---")
                        st.write(f"**Itens de teste (gostou):** {result.get('test_liked', [])}")
                        st.write(f"**Itens recomendados:** {result.get('recommended', [])}")
                        st.write(f"**Acertos (Hits):** {result.get('hits', 0)}")
                        
                except requests.RequestException as e:
                    st.error(f"Erro de conexão com o backend: {e}")

        st.markdown("---")
        st.subheader("Avaliação Geral do Modelo")
        
        if st.button("Calcular Médias Globais"):
            with st.spinner("Calculando acurácia para todos os usuários..."):
                try:
                    response = requests.get(f"{API_URL}/avaliar_acuracia_geral")
                    response.raise_for_status()
                    result = response.json()
                    
                    if "message" in result:
                        st.warning(result["message"])
                    else:
                        # --- MUDANÇA AQUI: Médias Globais ---
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Média Precision", f"{result.get('mean_precision', 0):.2%}")
                        c2.metric("Média Recall", f"{result.get('mean_recall', 0):.2%}")
                        c3.metric("Média F1", f"{result.get('mean_f1', 0):.2%}")
                        
                        st.write(f"**Total de usuários avaliados:** {result.get('users_evaluated', 0)}")
                except requests.RequestException as e:
                    st.error(f"Erro de conexão ou endpoint não encontrado: {e}")

def display_manga_details(item_id):
    """Renderiza a página de detalhes de um mangá específico."""
    global ratings_df
    if st.button(" Voltar ao Catálogo", key="back_button"):
        st.session_state.selected_manga_id = None
        st.rerun()

    selected_item = items_with_avg.loc[items_with_avg["item_id"] == item_id].iloc[0]
    st.header(selected_item["title"])
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(selected_item['image_url'], use_container_width=True)
    with col2:
        st.subheader("Detalhes")
        st.write(f"**Autor:** {selected_item['author']}")
        st.write(f"**Ano:** {selected_item['year']}")
        st.write(f"**Categoria:** {selected_item['category']}")
        st.write(f"**Média:** ⭐ {selected_item['avg_rating']:.2f}" if selected_item['avg_rating'] > 0 else "Sem avaliações")

    st.markdown("---")
    st.subheader("Sua Avaliação")
    current_user_id = st.number_input("Seu ID de usuário", min_value=1, step=1, value=st.session_state.current_user_id, key='user_id_input_detail')
    st.session_state.current_user_id = current_user_id
    
    user_rating_row = ratings_df[(ratings_df["user_id"] == current_user_id) & (ratings_df["item_id"] == item_id)]
    initial_rating = int(user_rating_row["rating"].iloc[0]) if not user_rating_row.empty else 3
    st.info(f"Sua avaliação atual: **{initial_rating}**." if not user_rating_row.empty else "Você ainda não avaliou este mangá.")
    
    new_rating = st.slider("Nota", 1, 5, initial_rating)
    if st.button("Salvar Minha Avaliação"):
        if not user_rating_row.empty:
            ratings_df.loc[user_rating_row.index, "rating"] = new_rating
            st.session_state.toast_message = {"message": "✅ Avaliação atualizada com sucesso!", "icon": "✅"}
        else:
            new_row = pd.DataFrame([{"user_id": current_user_id, "item_id": item_id, "rating": new_rating}])
            ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
            st.session_state.toast_message = {"message": "✅ Avaliação adicionada com sucesso!", "icon": "✅"}
        
        ratings_df.to_csv(RATINGS_CSV, index=False)
        st.cache_data.clear()
        st.rerun()

# --- Renderização Principal ---
if 'selected_manga_id' in st.session_state and st.session_state.selected_manga_id is not None:
    display_manga_details(st.session_state.selected_manga_id)
else:
    with st.sidebar:
        st.title("MangáRec")
        selected_page = option_menu(
            menu_title="Menu",
            options=["Catálogo", "Adicionar Avaliação", "Recomendações", "Acurácia"],
            icons=["collection", "plus-circle", "graph-up-arrow", "clipboard-data"],
            menu_icon="list-task",
            default_index=0,
        )
    
    if selected_page == "Catálogo":
        display_catalog()
    elif selected_page == "Adicionar Avaliação":
        display_add_rating()
    elif selected_page == "Recomendações":
        display_recommendations()
    elif selected_page == "Acurácia":
        display_accuracy()