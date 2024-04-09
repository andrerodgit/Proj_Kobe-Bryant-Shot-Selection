import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.ensemble import IsolationForest
import io
import matplotlib.image as mpimg

# Definir a URI de rastreamento do MLflow para apontar para o local desejado
mlflow.set_tracking_uri("../../data/modeling/tracking/mlruns")

# Pesquisar os experimentos disponíveis
experimentos = mlflow.search_experiments()

# Extrair os nomes dos experimentos
experimento_names = [exp.name for exp in experimentos]

# Exibir os experimentos no Streamlit
experimento_selecionado = st.sidebar.selectbox("Selecione o experimento", experimento_names, index=experimento_names.index("PipelineAplicacao"))

# Obter o ID do experimento selecionado
experimento_id = [exp.experiment_id for exp in experimentos if exp.name == experimento_selecionado][0]

# Carregar os dados de monitoramento do MLflow para o experimento selecionado
runs = mlflow.search_runs(experiment_ids=[experimento_id])

# Extrair eventos únicos relacionados ao status
eventos_arremessos = runs["status"].unique()

# Exibir os eventos de status no Streamlit
evento_arremesso_selecionado = st.sidebar.selectbox("Selecione o status do experimento", eventos_arremessos)

# Filtrar os dados por evento de arremesso selecionado
runs_filtered = runs[runs["status"] == evento_arremesso_selecionado].copy()  

# Ordenar o DataFrame pelo tempo de início da execução para garantir que o modelo mais recente esteja no topo
runs_filtered = runs_filtered.sort_values(by='start_time', ascending=False)

# Adicionar colunas ao DataFrame runs_filtered
runs_filtered["target"] = None
runs_filtered["best_model_type"] = None
runs_filtered["best_model"] = None
runs_filtered["model_features"] = None
runs_filtered["classification_threshold"] = None
runs_filtered["training_data_summary"] = None

# Obter as informações do modelo mais recente
latest_run_info = mlflow.get_run(runs_filtered.iloc[0]["run_id"]).data
latest_params = latest_run_info.params

# Preencher as colunas com os valores registrados no MLflow
runs_filtered.at[0, "target"] = latest_params.get("target")
runs_filtered.at[0, "best_model_type"] = latest_params.get("best_model_type")
runs_filtered.at[0, "best_model"] = latest_params.get("best_model")
runs_filtered.at[0, "model_features"] = latest_params.get("model_features")
runs_filtered.at[0, "classification_threshold"] = latest_params.get("classification_threshold")
runs_filtered.at[0, "training_data_summary"] = latest_params.get("training_data_summary")

# Criar um dashboard Streamlit para monitoramento do modelo
st.title("Dashboard de Monitoramento da operação")

# Exibir métricas de desempenho do modelo ao longo do tempo
st.subheader("Métricas de Desempenho do Modelo")
if not runs_filtered.empty:
    metrics_df = pd.DataFrame()  
    for col in runs_filtered.columns:
        if col.startswith("metrics."):  
            metric_name = col.split(".", 1)[1]  
            metrics_df[metric_name] = runs_filtered[col]  
    
    if not metrics_df.empty:  
        fig, axes = plt.subplots(len(metrics_df.columns), 1, figsize=(10, 5 * len(metrics_df.columns)))
        for i, column in enumerate(metrics_df.columns):
            ax = axes[i]
            ax.plot(metrics_df.index, metrics_df[column])
            ax.set_ylabel(column)
            ax.set_xlabel("Data")
            ax.set_title(f"Evolução da métrica {column} ao longo do tempo")
            ax.grid(True)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.subheader("Histogramas das Métricas")
        for column in metrics_df.columns:
            st.write(f"Histograma da métrica {column}")
            fig, ax = plt.subplots()
            sns.histplot(metrics_df[column], kde=True, ax=ax)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequência")
            st.pyplot(fig)
    else:
        st.write("Não há dados de métricas para exibir.")

# Exibir insights e alertas com base nas métricas
st.subheader("Insights e Alertas")
if not runs_filtered.empty:
    latest_metrics = runs_filtered.iloc[0]

    # Acessar as métricas corretamente
    log_loss_prod = latest_metrics.get("metrics.log_loss_prod")
    f1_score_prod = latest_metrics.get("metrics.f1_score_prod")

    # Log para depuração
    st.write(f"Log Loss: {log_loss_prod}")
    st.write(f"F1 Score: {f1_score_prod}")

    # Exibir as métricas se estiverem disponíveis
    if log_loss_prod is not None:
        st.write(f"Último Log Loss: {log_loss_prod}")
    if f1_score_prod is not None:
        st.write(f"Último F1 Score: {f1_score_prod}")

    # Definir os limiares para log loss e F1 score
    limiar_log_loss = 0.5  # Defina o limiar do log loss conforme necessário
    limiar_f1_score = 0.7  # Defina o limiar do F1 score conforme necessário

    # Log para depuração
    st.write(f"Limiar Log Loss: {limiar_log_loss}")
    st.write(f"Limiar F1 Score: {limiar_f1_score}")

    # Verificar se os valores estão fora dos limites aceitáveis e exibir alertas, se necessário
    if log_loss_prod is not None and log_loss_prod > limiar_log_loss:
        st.error("Desvio significativo detectado no log loss. Tome medidas adequadas.")
    if f1_score_prod is not None and f1_score_prod < limiar_f1_score:
        st.error("Desvio significativo detectado no F1 score. Tome medidas adequadas.")
else:
    st.write("Não há dados disponíveis para exibir insights e alertas.")

# Exibir informações do modelo e adicionais
#st.sidebar.write("Target: ", runs_filtered.iloc[0]["target"])
#st.sidebar.write("best_model_type: ", runs_filtered.iloc[0]["best_model_type"])
# st.sidebar.write("best_model: ", runs_filtered.iloc[0]["best_model"])
st.sidebar.write("limiar_log_loss: ", "0.5")
st.sidebar.write("limiar_f1_score: ", "0.7")

# Exibir os registros do MLflow
if not runs.empty:
    st.write("Número total de registros:", len(runs))
    st.write(runs)
else:
    st.write("Nenhum registro encontrado para o experimento PipelineAplicacao.")

# Exibir a matriz de confusão recuperada do MLflow
st.subheader("Matriz de Confusão")
if not runs_filtered.empty:
    # Obter o ID do run mais recente
    latest_run_id = runs_filtered.iloc[0]["run_id"]
    # Construir o caminho absoluto para a imagem da matriz de confusão
    artifact_path = "../../data/modeling/tracking/mlruns/456217708371445287/2c977f80eace48fa8b6aab39c22ac070/artifacts/confusion_matrix/confusion_matrix.png"
    try:
        # Carregar a imagem da matriz de confusão
        with open(artifact_path, "rb") as f:
            img_data = f.read()
        img = mpimg.imread(io.BytesIO(img_data), format='png')
        # Criar uma figura e eixos
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        # Exibir a figura usando st.pyplot()
        st.pyplot(fig)
    except FileNotFoundError:
        st.write("Não foi possível encontrar a imagem da matriz de confusão.")
else:
    st.write("Não há dados disponíveis para exibir a matriz de confusão.")

# Carregar as previsões do ambiente de desenvolvimento a partir do arquivo Parquet
development_predictions_path = "../../data/processed/train_holdout_predictions.parquet"
development_predictions_df = pd.read_parquet(development_predictions_path)

# Carregar as previsões do arquivo Parquet
predictions_path = "../../data/processed/production_predictions.parquet"
predictions_df = pd.read_parquet(predictions_path)


# Exibir distribuição das previsões do modelo ao longo do tempo
st.subheader("Distribuição das Previsões do Modelo")
if not predictions_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in predictions_df.columns:
        sns.kdeplot(development_predictions_df[column], label="Desenvolvimento", ax=ax, warn_singular=False)
    sns.kdeplot(predictions_df.stack(), label="Produção", ax=ax, warn_singular=False)
    # Adicionando manualmente as entradas da legenda para os dados de desenvolvimento e produção
    ax.legend(["Desenvolvimento", "Produção"])
    ax.set_xlabel("Valor da Saída do Modelo")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição das Previsões do Modelo em Desenvolvimento e Produção")
    st.pyplot(fig)
    
    # Estatísticas das previsões do modelo para a base de desenvolvimento
    st.write("Estatísticas das previsões do modelo para DEV:")
    st.write(development_predictions_df.describe())
    
    # Estatísticas das previsões do modelo para a nova base de produção
    st.write("\nEstatísticas das previsões do modelo para PRD:")
    st.write(predictions_df.describe())

    st.subheader("Histogramas das Previsões")
    for column in predictions_df.columns:
        st.write(f"Histograma da previsão {column}")
        fig, ax = plt.subplots()
        sns.histplot(predictions_df[column], kde=True, ax=ax, label="Produção")
        sns.histplot(development_predictions_df[column], kde=True, ax=ax, label="Desenvolvimento")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequência")
        ax.legend()
        st.pyplot(fig)

    # Estatísticas das previsões do modelo para a base de desenvolvimento
    print("Estatísticas das previsões do modelo para a base de desenvolvimento:")
    print(development_predictions_df.describe())
    
    # Estatísticas das previsões do modelo para a nova base de produção
    print("\nEstatísticas das previsões do modelo para a nova base de produção:")
    print(predictions_df.describe())
    
    # Detectar padrões incomuns nas previsões usando Isolation Forest
    st.subheader("Detecção de Padrões Incomuns nas Previsões")
    for column in predictions_df.columns:
        st.write(f"Análise de padrões incomuns na previsão {column}")
        # Preparar os dados para o Isolation Forest
        X_production = predictions_df[column].values.reshape(-1, 1)
        X_development = development_predictions_df[column].values.reshape(-1, 1)
        # Treinar o modelo Isolation Forest
        isolation_forest_production = IsolationForest(contamination=0.1)
        isolation_forest_production.fit(X_production)
        isolation_forest_development = IsolationForest(contamination=0.1)
        isolation_forest_development.fit(X_development)
        # Identificar anomalias
        anomalies_production = isolation_forest_production.predict(X_production)
        anomalies_development = isolation_forest_development.predict(X_development)
        anomalies_indices_production = [i for i, x in enumerate(anomalies_production) if x == -1]
        anomalies_indices_development = [i for i, x in enumerate(anomalies_development) if x == -1]
        st.write(f"Número de anomalias na produção: {len(anomalies_indices_production)}")
        st.write(f"Número de anomalias no desenvolvimento: {len(anomalies_indices_development)}")
        st.write(f"Índices das amostras anômalas na produção: {anomalies_indices_production}")
        st.write(f"Índices das amostras anômalas no desenvolvimento: {anomalies_indices_development}")
        # Plotar as previsões com as anomalias destacadas
        fig, ax = plt.subplots()
        ax.plot(predictions_df.index, predictions_df[column], label="Produção")
        ax.scatter(predictions_df.index[anomalies_indices_production], predictions_df[column][anomalies_indices_production], color='red', label='Anomalias Produção')
        ax.plot(development_predictions_df.index, development_predictions_df[column], label="Desenvolvimento")
        ax.scatter(development_predictions_df.index[anomalies_indices_development], development_predictions_df[column][anomalies_indices_development], color='green', label='Anomalias Desenvolvimento')
        ax.set_ylabel(column)
        ax.set_xlabel("Data")
        ax.set_title(f"Evolução da previsão {column} com anomalias destacadas")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.write("Não há dados de previsões para exibir.")