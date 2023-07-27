import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime

now = datetime.datetime.now()
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

## Dados de estudo ##
file_path = "/home/anisio/Documentos/Doutorado/Tese/analise/analise_others"+formatted_datetime+ " .csv"
path1 = "/home/anisio/Documentos/Doutorado/Tese"
## Dados de estudo ##
save_path = "/home/anisio/Documentos/Doutorado/Tese/Pesos"
path = "/home/anisio/Documentos/Doutorado/Tese/Dados"
dadosCampinaVerde = pd.read_csv(path + '/A519.csv',sep=';')
dadosSorriso = pd.read_csv(path + '/A904.csv',sep=';')
dadosDiamante = pd.read_csv(path + '/A849.csv',sep=';')
dadosCampo = pd.read_csv(path + '/A884.csv',sep=';')

listaDados = [dadosCampo,dadosSorriso,dadosDiamante,dadosCampinaVerde]

result = pd.read_csv(path1 + '/analise_2023-06-17 12:20:24 .csv',sep=',')

a = dadosCampo.describe()

listaDados = [dadosCampo,dadosSorriso,dadosDiamante,dadosCampinaVerde]
listNames = ["RS-CAMPOBOM","MT-SORRISO","PR-DIAMANTE DO NORTE","MG-CAMPINA VERDE"]

#############################################Correlação ########################################################
labels=['Umidade do Ar', 'Ponto de Orvalho', 'Pressão Atmosférica', 'Radiação Solar', 'Velocidade do Vento']
cont = 0

labels_dict = {
    'umid_inst': 'Umidade do Ar',
    'pto_orvalho_inst': 'Ponto de Orvalho',
    'pressao': 'Pressão Atmosférica',
    'radiacao': 'Radiação Solar',
    'vento_vel': 'Velocidade do Vento'
}
# Lista para armazenar os coeficientes de correlação

correlation_values = []
# Criar a figura e os subplots
#fig, axs = plt.subplots(nrows=len(listaDados), ncols=1, figsize=(8, 4*len(listaDados)), sharex=True)

fig, axs = plt.subplots(nrows=int(np.ceil(len(listaDados) / 2)), ncols=2, figsize=(16, 4*len(listaDados)), sharex=True)
for i, df in enumerate(listaDados):
    df = df[['data','hora', 'umid_inst', 'pto_orvalho_inst', 'pressao','radiacao', 'vento_vel','temp_inst']]
    
    
    # Converter a coluna 'hora' para string
    df['hora'] = df['hora'].astype(str)
    df['hora_'] = df['hora'].apply(lambda x: x.zfill(4))
    
    
    # Juntar os campos 'data' e 'hora' em um único campo 'data_hora' do tipo datetime
    #df['data_'] = # Juntar os campos 'data' e 'hora' em um único campo 'data_hora' do tipo datetime
    df['data_hora'] = pd.to_datetime(df['data'] + '-' + df['hora_'], format='%Y-%m-%d-%H%M')
    
    # Converter a coluna 'hora' para o tipo 'int'
    df['hora'] = df['hora'].astype(int)
    
    # Converter as colunas restantes para o tipo 'float'
    colunas_float = ['pressao', 'radiacao', 'temp_inst', 'umid_inst', 'vento_vel']
    df[colunas_float] = df[colunas_float].astype(float)
    
    df = df.dropna()
    df = df[['umid_inst', 'pto_orvalho_inst', 'pressao','radiacao', 'vento_vel','temp_inst']]
    
    # Calculate the correlation coefficients
    correlation = df.corr()['temp_inst'].drop('temp_inst')
    correlation_values.append(correlation.values)
    
    # Criar DataFrame temporário para o gráfico de barras
    df_temp = pd.DataFrame({'Variáveis Independentes': labels, 'Coeficiente de Correlação': correlation_values[i]})
    
    
    # Set the color palette to cool colors
    colors = sns.color_palette("cool", len(correlation))
    
    # Definir cores personalizadas
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    # Criar o gráfico de barras com as cores atribuídas
    sns.barplot(ax=axs[i//2, i%2], data=df_temp, x='Variáveis Independentes', y='Coeficiente de Correlação', palette=colors)

    # Personalizar os rótulos dos eixos, o título e o espaçamento
    axs[i//2, i%2].set_ylabel('Coeficiente de Correlação', fontsize=12)
    axs[i//2, i%2].set_title("{}".format(listNames[i]), fontsize=14, fontweight='bold')

    # Remover rótulos do eixo x
    axs[i//2, i%2].set_xticks([])
    axs[i//2, i%2].set_xlabel('')

# Exibir a legenda com os nomes dos itens em listaDados fora do último subplot
#axs[-1, 0].legend(title='Item', loc='upper left', labels=listNames)

# Personalizar xlabel para todos os subplots
fig.text(0.5, -0.02, 'Variáveis Independentes', ha='center', fontsize=24,fontweight='bold')
fig.text(0.5, 1, 'Correlação com a Temperatura do Ar', ha='center', fontsize=24,fontweight='bold')

# Ajustar espaçamento entre subplots
plt.tight_layout()
# Adicionar legenda ao gráfico
for idx, label in enumerate(correlation.index):
    plt.bar(0, 0, color=colors[idx], label=labels[idx])
# Remover as bordas do gráfico
# Configurar a legenda para ser exibida fora das imagens combinadas
#legend = axs[0, 0].legend(title='Item', loc='upper center', labels=listNames, bbox_to_anchor=(0.5, 1.1), ncol=len(listaDados)//2)
#legend = axs[0, 0].legend(title='Item', loc='upper center', labels=listNames, bbox_to_anchor=(0.5, -0.1), ncol=len(listaDados)//2, frameon=True, facecolor='white', edgecolor='black')
#legend = axs[0, 0].legend(title='Item', loc='upper center', labels=listNames, bbox_to_anchor=(0.5, 1.15), ncol=len(listaDados)//2, frameon=True, facecolor='white', edgecolor='black')
# Exibir a legenda com os nomes dos eixos x
plt.legend(title='Variáveis Independentes', loc='upper left', bbox_to_anchor=(1.05, 1.2),fontsize=14)



# Exibir a imagem com os subplots
plt.show()
plt.savefig('correlation_plots.png', dpi=300)
    