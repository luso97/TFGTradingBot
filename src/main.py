'''
Created on 19 may. 2020

@author: Luis
'''
import pandas as pd
import tensorflow as tf
import bs4 as bs
import requests
import matplotlib.pyplot as plt
import os
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import time;
import math
from hurst import compute_Hc, random_walk
from sklearn.model_selection import cross_val_score
import alpaca_trade_api as tradeapi
from src.historicData import createTradeDfs, getSandP500,createDataFrameHistories, getAllStocks,selectStocks
from src.Strategy import Estrategia
from src.model import train_model
from datetime import date
import schedule


load=True;
#load=True
histories={}
stocks=getAllStocks();
symbols = selectStocks();
dfs = {}
accNum = 7;
if load == False:
    for sym in symbols:
        print(sym)
        
        df = createTradeDfs(sym);
        dfs[sym]=df;
        hist = train_model(df)
        histories[sym]=hist;
    d = createDataFrameHistories(histories)
    dfRank = pd.DataFrame.from_dict(d,orient="index")
    dfRank['Clases 0x1']=(dfRank['Clase 0']+dfRank['Clase 2'])/2
    acciones = dfRank.sort_values(by="Clases 0x1",ascending=False).head(accNum).index
    dfs2={}
    print(acciones);
    for sym in acciones:
        dfs2[sym]=dfs[sym];
else:
    symbols=[stocks[0]]
    for sym in symbols:
        df = createTradeDfs(sym);
        dfs[sym]=df;
    acciones = [stocks[0]]
    dfs2={}
    for sym in acciones:
        dfs2[sym]=dfs[sym];
estrategia = Estrategia(acciones,dfs2,20,0.02,0.03,100,5);
if load==True:
    estrategia.load()
schedule.every(50).seconds.do(estrategia.getStatus)
schedule.every().day.at("15:00").do(estrategia.signalOrders);
schedule.every().day.at("15:05").do(estrategia.addFilas);
while True:
    schedule.run_pending();
#get 

#Primero entrenamos nuestro modelo con diferentes acciones