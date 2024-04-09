import os
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import confusion_matrix, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from pycaret.classification import setup, create_model, predict_model
import matplotlib.pyplot as plt
import numpy as np

# Definir a URI de rastreamento do MLflow para apontar para o local desejado
mlflow.set_tracking_uri("../../data/modeling/tracking/mlruns")

# Definir o experimento onde a execução será registrada
mlflow.set_experiment("PipelineAplicacao")

# Função para recomendar estratégias de arremesso com base nas condições de jogo
def recomendar_estrategias_arremesso(condicoes_de_jogo, df_prod):
    # Aqui você pode implementar a lógica para recomendar estratégias de arremesso com base nas condições de jogo
    # Por enquanto, vamos apenas retornar os primeiros 5 registros do DataFrame de produção
    return df_prod.head(5)

# Função para recomendar os 5 melhores momentos de arremesso
def recomendar_melhores_momentos_arremesso(df_prod, y_prod_pred_proba):
    # Ordenar os dados com base nas previsões de probabilidade
    df_prod['predicted_proba'] = y_prod_pred_proba
    df_prod_sorted = df_prod.sort_values(by='predicted_proba', ascending=False)
    
    # Selecionar os 5 melhores momentos de arremesso
    melhores_momentos = df_prod_sorted.head(5)
    return melhores_momentos

# Iniciar a execução do MLflow
with mlflow.start_run(run_name="PipelineAplicacao"):
    # Carregar a base de produção
    prod_data_path = "../../data/raw/dataset_kobe_prod.parquet"
    df_prod = pd.read_parquet(prod_data_path)

    # Selecionar apenas as colunas especificadas
    selected_columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    df_prod = df_prod[selected_columns]

    # Preparar os dados de entrada
    X_prod = df_prod.drop('shot_made_flag', axis=1)
    y_prod_true = df_prod['shot_made_flag']

    # Etapa 1: Treinamento do modelo com regressão logística do pycaret
    clf = setup(data=df_prod, target='shot_made_flag', verbose=False)
    model = create_model('lr')
    
    # Adicionar um print para identificar o modelo escolhido
    # print("Modelo escolhido:", model)

    # Etapa 2: Registro do modelo com threshold de precisão mínimo de 70% em Staging
    pred_holdout = predict_model(model)
    mlflow.sklearn.log_model(model, "model", registered_model_name="Modelo_Kobe_Bryant", input_example=X_prod.head())
    
    # Salvar os resultados de holdout predictions como um arquivo parquet
    pred_holdout_path = "../../data/processed/holdout_predictions.parquet"
    pred_holdout.to_parquet(pred_holdout_path, index=False)
    mlflow.log_artifact(pred_holdout_path, artifact_path="holdout_predictions")

    # Etapa 3: Aplicação Online: recomendação de arremessos
    # Definir as condições de jogo
    condicoes_de_jogo = {
        'tempo_restante': 5,  # exemplo de parâmetro
        'pontuacao': 90,  # exemplo de parâmetro
        # outros parâmetros necessários...
    }
    
    # Chamada da função com os parâmetros definidos
    estrategias_recomendadas = recomendar_estrategias_arremesso(condicoes_de_jogo, df_prod)
    print("Estratégias de arremesso recomendadas: ")
    print(estrategias_recomendadas)

    # Etapa 4: Propor os 5 melhores momentos de arremesso
    predicted_proba_col = predict_model(model, data=X_prod)
    
    # Acessar a coluna de probabilidade prevista corretamente
    prob_column_name = predicted_proba_col.columns[-1]  # Última coluna do DataFrame
    y_prod_pred_proba = predicted_proba_col[prob_column_name]  # Usar o nome da última coluna
    melhores_momentos = recomendar_melhores_momentos_arremesso(df_prod, y_prod_pred_proba)
    print("Os 5 melhores momentos de arremesso recomendados: ")
    print(melhores_momentos)

    # Etapa 5: Aplicação de Monitoramento
    # Como exemplo, vamos calcular as métricas de desempenho do modelo na base de produção
    log_loss_prod = log_loss(y_prod_true, y_prod_pred_proba)
    f1_score_prod = f1_score(y_prod_true, (y_prod_pred_proba > 0.5).astype(int))

    # Definir os limiares para log loss e F1 score
    limiar_log_loss = 0.5
    limiar_f1_score = 0.7

    # Comparar as métricas com limiares predefinidos e tomar medidas adequadas, se necessário
    if log_loss_prod > limiar_log_loss:
        print("Desvio significativo detectado no log loss. Tome medidas adequadas.")
    if f1_score_prod < limiar_f1_score:
        print("Desvio significativo detectado no F1 score. Tome medidas adequadas.")

    # Registrar métricas de desempenho no MLflow
    mlflow.log_metric("log_loss_prod", log_loss_prod)
    mlflow.log_metric("f1_score_prod", f1_score_prod)

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_prod_true, (y_prod_pred_proba > 0.5).astype(int))

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Reais')
    tick_marks = np.arange(len(np.unique(y_prod_true)))
    plt.xticks(tick_marks, ['Classe Negativa', 'Classe Positiva'])
    plt.yticks(tick_marks, ['Classe Negativa', 'Classe Positiva'])
    thresh = cm.max() / 2.
    for i, j in [(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    # Obter o ID do experimento atual
    experiment_id = mlflow.active_run().info.experiment_id

    # Construir o caminho completo para a pasta de artefatos do experimento
    artifacts_path = "../../data/modeling/tracking/mlruns/456217708371445287/"

    # Criar a pasta de artefatos se não existir
    os.makedirs(artifacts_path, exist_ok=True)

    # Salvar a imagem da matriz de confusão na pasta de artefatos do MLflow
    confusion_matrix_path = os.path.join(artifacts_path, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
        
    # Registrar a imagem no MLflow
    mlflow.log_artifact(confusion_matrix_path, artifact_path="confusion_matrix")

  
# Criar DataFrame com as previsões na produção
production_predictions = predict_model(model, data=X_prod)
#print(production_predictions.head())
#print(production_predictions.columns)

# Atualizar o nome da coluna de acordo com as previsões
prediction_column_name = 'prediction_label'  # ou 'prediction_score' dependendo da coluna correta

# Criar o DataFrame de produção com as colunas corretas
production_predictions = pd.DataFrame({'Actual': y_prod_true, 'Predicted_LR': production_predictions[prediction_column_name], 'Predicted_Probability': y_prod_pred_proba})
     #  predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted_LR': y_pred_lr, 'Predicted_Probability': y_pred_dt})

# Salvar as previsões na produção como arquivo Parquet
production_predictions_path = "../../data/processed/production_predictions.parquet"
production_predictions.to_parquet(production_predictions_path, index=False)
mlflow.log_artifact(production_predictions_path, artifact_path="production_predictions")
