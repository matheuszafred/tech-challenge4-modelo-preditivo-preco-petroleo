import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('Dashboard - Variação Preço Petroleo Brent')


# URL da tabela de preços do petróleo Brent
url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

# Lendo os dados diretamente da tabela HTML
dfs = pd.read_html(url, match='Data')
df = dfs[1]
df.head()

# Definindo a primeira linha como cabeçalho
df.columns = df.iloc[0]  # Define a primeira linha como cabeçalho
df = df[1:]  # Remove a primeira linha do DataFrame
df.head()

# Renomeando colunas
df.columns = ['Data', 'Preco']
df.head()

# Convertendo a coluna de data para datetime e o preço para float
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
df['Preco'] = df['Preco'].str.replace(',', '.').astype(float)

df.info()

# Definindo a data como índice
df.set_index('Data', inplace=True)
df.head()

st.markdown(' ### Big Numbers')

# Visualização dos visuais no Streamlit
# Cartões
cor_estilizada = 'color: #0145AC;'
fonte_negrito = 'font-weight: bold;'

col1, col2, col3, col4 = st.columns(4)
with col1:
    metrica1 = df.index.max().strftime('%d/%m/%Y')
    st.markdown(f"<h2 style='{cor_estilizada}'>{metrica1}</h2> <span style='{
                fonte_negrito}'> Dados atualizados até </span>", unsafe_allow_html=True)
    # st.metric('Dados atualizados até:', value=dados.index.max().strftime('%d/%m/%Y'))
with col2:
    metrica2 = df.index.min().strftime('%d/%m/%Y')
    st.markdown(f"<h2 style='{cor_estilizada}'> {metrica2} </h2> <span style='{
                fonte_negrito}'> Dados monitorados desde</span> ", unsafe_allow_html=True)
with col3:
    metrica3 = df['Preco'].min()
    data_metrica3 = df[df['Preco'] == df['Preco'].min()].index
    st.markdown(f"<h2 style='{cor_estilizada}'> US$ {metrica3:.2f} </h2> <span style='{
                fonte_negrito}'> Menor preço histórico <br> (atingido em  {data_metrica3[0].strftime('%d/%m/%Y')})</span> ", unsafe_allow_html=True)
    # st.metric('Menor preço histórico:', value=dados['Preco'].min().round(2))
with col4:
    metrica4 = df['Preco'].max()
    data_metrica4 = df[df['Preco'] == df['Preco'].max()].index
    st.markdown(f"<h2 style='{cor_estilizada}'> US$ {metrica4:.2f} </h2> <span style='{
                fonte_negrito}'> Maior preço histórico <br> (atingido em  {data_metrica4[0].strftime('%d/%m/%Y')})</span> ", unsafe_allow_html=True)


st.markdown(' ### Análise e criação de insights')

df.describe()

st.dataframe(df.describe())

texto = """

**Análise estatistica:**

Esse comportamento estatístico reflete um mercado historicamente volátil, influenciado por fatores como crises econômicas, conflitos globais e mudanças na demanda energética.


"""

st.write(texto)

# Plotando os preços do petróleo
plt.figure(figsize=(18, 7))
plt.plot(df.index, df['Preco'], label='Preço do Petróleo Brent', color='blue')
plt.title('Preço do Petróleo Brent ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Preço em USD')
plt.grid()
plt.legend()
plt.show()

st.pyplot(plt)

texto = """

**Comportamento Temporal**

1. Tendências de Longo Prazo:

- Observa-se um crescimento acentuado nos preços entre os anos 2000 e 2008, possivelmente relacionado ao aumento da demanda global e eventos geopolíticos.
Após 2008, houve uma queda brusca, possivelmente devido à crise financeira global.
Períodos subsequentes mostram flutuações cíclicas, com picos significativos em 2012-2014 e uma queda acentuada em 2015-2016, ligada à superprodução global e ao uso crescente de fontes de energia alternativas.

2. Recente Volatilidade:

- Após 2020, observa-se uma recuperação após a pandemia de COVID-19, mas os preços ainda mostram um padrão volátil, possivelmente devido à incerteza global e questões como a guerra na Ucrânia.

"""

st.write(texto)


st.markdown(' ### Modelo Preditivo ETS - Preço do Petroleo Brent')

# Ordenando o DataFrame em ordem cronológica crescente
df = df.sort_index(ascending=True)

# Verificando a última data após a ordenação
ultima_data = df.index.max()  # Última data nos dados
print(f"Última data no DataFrame: {ultima_data}")

# Configurações para previsão
qt_dias_historico = 180  # Usar os últimos 180 dias
qt_dias_prever = 30  # Prever os próximos 30 dias
trend = 'add'  # Componente de tendência
seasonal = 'add'  # Componente de sazonalidade


def modelo_ets_previsao(dados, qt_dias_historico, qt_dias_prever, trend, seasonal):
    # Verificar se os dados estão vazios
    if dados.empty:
        raise ValueError("Os dados fornecidos estão vazios.")

    # Filtrar apenas os últimos qt_dias_historico dias
    if len(dados) < qt_dias_historico:
        raise ValueError(f"Os dados fornecidos contêm apenas {len(dados)} dias, mas {
                         qt_dias_historico} dias são necessários.")

    # Selecionando os últimos 'qt_dias_historico' registros para treinamento
    dados_treino = dados.tail(qt_dias_historico)

    # Criando o modelo ETS
    modelo_ets = ExponentialSmoothing(
        dados_treino['Preco'], trend=trend, seasonal=seasonal, seasonal_periods=30)

    # Treinando o modelo
    resultado = modelo_ets.fit()

    # Fazendo previsões
    previsao = resultado.forecast(steps=qt_dias_prever)

    return previsao


# Fazendo a previsão
previsao_ets = modelo_ets_previsao(
    df, qt_dias_historico, qt_dias_prever, trend, seasonal)

# Gerando as datas futuras corretamente
future_dates = pd.date_range(
    start=ultima_data + pd.Timedelta(days=1), periods=qt_dias_prever, freq='D')

# Exibindo informações para verificar a consistência
print(f"Datas futuras geradas: {future_dates}")

# Plotando os dados históricos e a previsão
plt.figure(figsize=(16, 7))

# Plotando os últimos 180 dias de dados históricos
plt.plot(df[-qt_dias_historico:].index, df['Preco']
         [-qt_dias_historico:], label='Dados Históricos', color='blue')

# Plotando as previsões
plt.plot(future_dates, previsao_ets, label='Previsão ETS', color='red')

# Configurações do gráfico
plt.title('Previsão do Preço do Petróleo Brent com ETS')
plt.xlabel('Data')
plt.ylabel('Preço em USD')
plt.legend()
plt.grid()
plt.show()

st.pyplot(plt)

texto = """

**Análise do Modelo Preditivo ETS**

1. Previsão de Tendência:

- O modelo ETS projeta uma leve estabilização no preço do petróleo Brent para o final de 2024, com os preços oscilando na faixa de 72 a 76 USD.
A previsão sugere uma redução na volatilidade em comparação aos meses anteriores.

2. Consistência com Dados Históricos:

- O modelo captura bem a sazonalidade e flutuações observadas no histórico recente, indicando que as variáveis tendência e sazonalidade são fatores predominantes nos preços do Brent.

**Insights Adicionais para Tomada de Decisão**

1. Estratégia de Compras e Estoques:

- Com base na previsão de estabilização, empresas dependentes do petróleo podem planejar compras para os próximos meses sem grandes riscos de aumento abrupto nos preços.
- Estratégias de estoque podem ser ajustadas para maximizar o custo-benefício no curto prazo.

2. Planejamento Financeiro e Orçamentário:

- Organizações que utilizam o petróleo Brent como insumo (ex.: empresas de transporte e manufatura) podem usar a previsão para alinhar custos e precificar produtos com maior previsibilidade.

3. Avaliação de Riscos:

- Embora o modelo ETS projete estabilidade, a análise deve incluir um monitoramento contínuo de eventos externos, como a guerra na Ucrânia ou decisões da OPEP, que podem alterar significativamente os preços.

4. Cenário de Investimento:

- Para investidores no mercado de petróleo, a previsão de estabilidade pode sugerir menor atratividade para especulação de curto prazo, mas boas oportunidades para investimentos seguros ou hedge.


"""

st.write(texto)
