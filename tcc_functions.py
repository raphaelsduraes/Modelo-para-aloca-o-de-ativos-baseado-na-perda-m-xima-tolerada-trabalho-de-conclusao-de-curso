import quandl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.ticker import PercentFormatter

def calculo_retorno_ativos(precos):
    retorno = (precos/precos.shift(1))-1 
    return retorno

def log_retorno(retorno_linear): return np.log(1+retorno_linear)

def retornos(data):# no formato 'AAAA-MM-DD'
    """Entre com a data no formato 'AAAA-MM-DD'
    
    Retorna um DataFrame com o retorno diário em porcentagem 
    desde a data imposta até os dias atuais do dólar comercial,
    índice IBOVESPA e CDI
    """
    data_inicial = data
   
    """Leitura dos preços diários dos índices do S&P 500, Ibovespa 
    e preços do dólar usandos as plataformas Quandl e Yahoo Finance"""
    
    dolar_diario_preco_compra = quandl.get("BCB/10813",
                                           authtoken="wHnVchHSPwZYMsXTPAUm",
                                           start_date = data_inicial)
    
    ibovespa_pontos_ajustados =quandl.get("BCB/7",
                                          authtoken="wHnVchHSPwZYMsXTPAUm",
                                          start_date = data_inicial)
    
    #Variáveis com retorno diário em porcentagem
    cdi_diario = quandl.get("BCB/12",
                            authtoken="wHnVchHSPwZYMsXTPAUm",
                            start_date = data_inicial)/100
    
    dolar_diario = calculo_retorno_ativos(dolar_diario_preco_compra)
    ibovespa = calculo_retorno_ativos(ibovespa_pontos_ajustados)
    
    # Criar DataFrame com os retornos e nomes nas colunas
    retorno_diario = pd.DataFrame(dolar_diario)
    ativos = [cdi_diario, ibovespa]
    for ativo in ativos:
        ativo = pd.DataFrame(ativo)
        retorno_diario = retorno_diario.merge(ativo,
                                              left_on='Date',
                                              right_on='Date')
    retorno_diario=retorno_diario.rename(columns={'Value_x':'retorno_dolar_m',
                                                  'Value_y':'retorno_cdi_m',
                                                  'Value':'retorno_ibovespa_m'})
    
    retorno_diario = retorno_diario.iloc[1:]
    retorno_diario = retorno_diario.fillna(0)
    #retorno_diario.index = pd.to_datetime(ativos.index, format="%Y-%m-%d")
    def retorno_mensal(ativo):
    # Usar o método de log_retorno onde o retorno acumulado basta somar
        log_ret = log_retorno(ativo)
        log_retorno_mensal=log_ret.resample('M').sum()
        # transformar o log retorno em retorno linear
        retorno_linear_mensal = np.e**log_retorno_mensal - 1
        return retorno_linear_mensal
    
    retorno_m = retorno_mensal(retorno_diario)
    return retorno_m

def retorno_acumulado(ativo):
    ativo = ativo + 1
    retornos_acumulados = ativo.cumprod()
    return (retornos_acumulados - 1)*100 

def assimetria(r):
    """
    Semelhante ao comando scipy.stats.skew()
    Calcula a assimetria de uma distribuicao
    Retorna um numero float ou uma serie
    """
    r_menos_r_medio = r - r.mean()
    # use the population standard deviation, so set dof=0
    desvio_padrao = r.std(ddof=0)
    esperanca = (r_menos_r_medio**3).mean()
    return esperanca / desvio_padrao**3


def curtose(r):
    """
    Calcula a curtose de uma distribuicao
    Retorna um numero float ou uma serie
    """
    r_menos_r_medio = r - r.mean()
    # Use o desvio padrao da populacao, usar o ddof=0 no metodo .std
    desvio_padrao = r.std(ddof=0)
    esperanca = (r_menos_r_medio**4).mean()
    return esperanca/desvio_padrao**4

import scipy.stats
def jarque_bera(r, nivel = 0.01):
    """
    Aplica o teste de Jarque-Bera para achar se uma série segue uma distribuiçao normal com  1% de confiança
    Retorna True se a hipótese de normalidade for aceita e falso se a hipótese não for aceita
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(jarque_bera)
    else:
        statistic, p_valor = scipy.stats.jarque_bera(r)
        return p_valor> nivel

def sumario_estatistico(ativos):
    sumario =  pd.concat([ativos.mean(), 
                                      ativos.median(),
                                      ativos.mean()>ativos.median(),
                                      ativos.max(),
                                      ativos.min(),
                                      ativos.std(),
                                      assimetria(ativos),
                                      curtose(ativos),
                                      jarque_bera(ativos)],
                                      axis=1,)
    nome_estatisticas = ['media','mediana','media>mediana?','máxima','mínima','desvio padrao','assimetria','curtose','teste de jarque_bera']
    sumario.columns = nome_estatisticas
    return sumario
def var_historico(r, level=1):
    """
    Retorna o valor em risco histórico de acordo nível desejado
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historico, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("A variável de entrada deve ser uma série ou um DataFrame")
        
from scipy.stats import norm
def var_gaussiano(r, level=1):
    """
    Retorna o valor em risco paramétrico gaussiano de uma série ou DataFrame
    """
    # Use o número Z assumindo que segue uma distribuição normal
    z = norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))

from scipy.stats import norm
def var_cornish_fisher(r, level=1, modified=False):
    """
    Retorna o Valor em risco paramétrico gaussiano
    usando a correção de cornish-Fisher """
    # Usa o valor de Z considerando gaussiano 
    z = norm.ppf(level/100)
    
    # método que modifica o valor de Z usando a curtose e assimetria
    c = curtose(r)
    a = assimetria(r)
    z = (z +
            (z**2 - 1)*a/6 +
            (z**3 -3*z)*(c-3)/24 -
            (2*z**3 - 5*z)*(a**2)/36
            )
        
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historico(r, level=1):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historico(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historico, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
def sumario_var(ativos):
    tabela_var = [var_historico(ativos), 
                  var_gaussiano(ativos), 
                  var_cornish_fisher(ativos)]
    comparacao = pd.concat(tabela_var,
                           axis=1)
    comparacao.columns=['histórico',
                        'Gaussiano',
                        'Cornish-Fisher']
    
    return comparacao.plot.bar(title="Valor em Risco: Dólar, CDI, e Ibovespa, nível de confiança de 1%",
                        figsize=(10,6))

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return np.matmul(weights,returns)

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (np.matmul(np.matmul(weights.T,covmat), weights))**0.5

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax
    
def pesos():
    lista=[]
    for a in range(0,1000):
        if a/200<=1:
           lista.append(a/200)

    p_dolar=[]
    p_cdi=[]
    p_ibovespa=[]
    for a in lista:
        for b in lista:
            for c in lista:
                if a + b + c == 1:
                    p_dolar.append(a)
                    p_cdi.append(b)
                    p_ibovespa.append(c)
    peso = [p_dolar, p_cdi, p_ibovespa]
    data =[]
    for i in peso:
        data.append(pd.DataFrame(i))
    pesos = pd.concat([data[0] ,data[1], data[2]], axis=1, sort=False)
    pesos.columns=['p_dolar','p_cdi','p_ibovespa']
    return pesos

def lista_peso():    
    peso = pesos()
    lista_peso=[]
    for i in range(0,len(peso)):
        lista_peso.append(np.array(peso.iloc[i]))
    return pd.DataFrame(lista_peso)

def carteiras_peso_retorno_volatilidade(er, cov):    
    peso_ativo = lista_peso().values
    retornos = [portfolio_return(w, er.values) for w in peso_ativo]
    vols = [portfolio_vol(w, cov.values) for w in peso_ativo]
    peso_ativo = pd.DataFrame(peso_ativo)
    retornos = pd.DataFrame(retornos)
    vols = pd.DataFrame(vols)
    
    # anualizar retorno e volatilidade
    retornos = ((retornos + 1)**12) - 1
    vols = vols*(12**0.5)
    
    # juntar as séries
    tabela = pd.concat([peso_ativo,vols, retornos],
                       axis=1, sort=False)
    
    tabela.columns = ['peso_dolar',
                      'peso_cdi',                      
                      'peso_ibovespa',
                      'vol_carteira','ret_carteira']
    
    tabela = tabela.round(6)
    tabela = tabela.sort_values(by='vol_carteira')
    def novo_indice(tabela):
        novo_indice=[]
        for i in range (0,tabela.shape[0]):
            novo_indice.append(i)
        novo_indice[-1], tabela.shape[0]
        return novo_indice

    novo_ind = novo_indice(tabela)
    tabela.index = novo_ind
    return tabela

def fronteira_eficiente(tabela):
        fronteira_eficiente = tabela.round(2)
        fronteira_eficiente = pd.DataFrame(fronteira_eficiente.groupby(['vol_carteira'])['ret_carteira'].max())
        fronteira_eficiente.sort_values(by ='ret_carteira')
        fronteira_eficiente['Volatilidade'] = fronteira_eficiente.index
        fronteira_eficiente.columns=['Retorno anual','Volatilidade']
        return fronteira_eficiente

def carteiras_eficientes(tabela):
        fronteira_eficiente = tabela.round(2)
        carteiras_eficientes = pd.DataFrame(fronteira_eficiente.groupby(['vol_carteira'])['ret_carteira'].idxmax())
        carteiras_eficientes = carteiras_eficientes['ret_carteira'].tolist()
        return tabela.iloc[carteiras_eficientes]
    
def grafico_carteiras(retorno_medio,covariancia):
    tabela = carteiras_peso_retorno_volatilidade(retorno_medio
                                           ,covariancia)
    dolar_ponto = [tabela[tabela['peso_dolar']==1]['vol_carteira'],
                   tabela[tabela['peso_dolar']==1]['ret_carteira']]

    cdi_ponto = [tabela[tabela['peso_cdi']==1]['vol_carteira'],
                 tabela[tabela['peso_cdi']==1]['ret_carteira']]

    ibovespa_ponto = [tabela[tabela['peso_ibovespa']==1]['vol_carteira'],
                      tabela[tabela['peso_ibovespa']==1]['ret_carteira']]

    front_eficiente = fronteira_eficiente(tabela)
    return (
    #Todas as carteiras possíveis com o andamento de 0,05%
    tabela.plot.scatter(x='vol_carteira',
                        y='ret_carteira',
                        label='Carteiras não eficientes',
                        figsize=(12,6),fontsize='xx-large'),

    # Risco e retorno dos ativos
    plt.plot(dolar_ponto[0],
             dolar_ponto[1],
             label='Dólar',
             marker='o',
             color='green',
             markersize=12),

    plt.plot(cdi_ponto[0],
             cdi_ponto[1],
             label='CDI',
             marker='o',
             color='red',
             markersize=12),

    plt.plot(ibovespa_ponto[0],ibovespa_ponto[1],
             label='Ibovespa',
             marker='o',
             color='black',
             markersize=12),

    #Fronteira eficiente
    plt.plot(front_eficiente.index,
             front_eficiente['Retorno anual'],color='brown',label='Carteiras eficientes'),
    
    plt.xlabel('Risco (desvio-padrão)',fontsize='xx-large'),
    plt.ylabel('Retorno anual',fontsize='xx-large'),
    plt.legend(fontsize='xx-large'),
    )


def retorno_carteiras_eficientes(ativos,carteiras_eficientes):
    carteiras = ativos
    for i in range(0,carteiras_eficientes.shape[0]):
        carteira_i_eficiente=pd.DataFrame(np.matmul(ativos.values,carteiras_eficientes.iloc[i,0:3].values),
                                          index=ativos.index,
                                          columns=['carteira %i' %(i+1)])
        carteiras=carteiras.join(carteira_i_eficiente)
    return carteiras

def grafico_retorno_acumulado(carteiras):
    grafico_carteiras = retorno_acumulado(carteiras)
    grafico_carteiras = grafico_carteiras.plot(figsize=(10,6),fontsize='xx-large')
    grafico_carteiras.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,fontsize='large')
    grafico_carteiras.yaxis.set_major_formatter(PercentFormatter())

    return grafico_carteiras

def grafico_var(tabela_var,i ='vol_carteira'):
    """ Calcular o gráfico de rentabilidade em função do desvio-padrão ou Var de Cornish-Fisher ou 
     ou Conditional Var 
     entre com o i = 'vol_carteira', ou 'var_cornish_fisher' ou 'cvar' """
    tabela = tabela_var
    dolar_ponto = [tabela[tabela['peso_dolar']==1][i],
                   tabela[tabela['peso_dolar']==1]['ret_carteira']]

    cdi_ponto = [tabela[tabela['peso_cdi']==1][i],
                 tabela[tabela['peso_cdi']==1]['ret_carteira']]

    ibovespa_ponto = [tabela[tabela['peso_ibovespa']==1][i],
                      tabela[tabela['peso_ibovespa']==1]['ret_carteira']]

    def fronteira_eficiente(tabela):
        fronteira_eficiente = tabela.round(2)
        fronteira_eficiente = pd.DataFrame(fronteira_eficiente.groupby([i])['ret_carteira'].max())
        fronteira_eficiente.sort_values(by='ret_carteira')
        fronteira_eficiente['Valor']=fronteira_eficiente.index
        fronteira_eficiente.columns=['Retorno anual',i]
        return fronteira_eficiente
    front_eficiente = fronteira_eficiente(tabela)
    return (
    #Todas as carteiras possíveis com o andamento de 0,05%
    tabela.plot.scatter(x=i,
                        y='ret_carteira',
                        label='Carteiras não eficientes',
                        figsize=(12,6)),

    # Risco e retorno dos ativos
    plt.plot(dolar_ponto[0],
             dolar_ponto[1],
             label='Dólar',
             marker='o',
             color='green',
             markersize=12),

    plt.plot(cdi_ponto[0],
             cdi_ponto[1],
             label='CDI',
             marker='o',
             color='red',
             markersize=12),

    plt.plot(ibovespa_ponto[0],ibovespa_ponto[1],
             label='Ibovespa',
             marker='o',
             color='black',
             markersize=12),

    #Fronteira eficiente
    plt.plot(front_eficiente.index,
             front_eficiente['Retorno anual'],color='brown',label='Carteiras eficientes'),
    
    plt.xlabel('Risco('+ i +')', fontsize='xx-large'),
    plt.ylabel('Retorno anual', fontsize='xx-large'),
    plt.legend(fontsize='xx-large'),
    )

