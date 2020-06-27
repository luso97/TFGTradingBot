'''
Created on 19 may. 2020

@author: Luis
'''
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import bs4 as bs
import requests
import time;
import numpy as np;
import os;

alphaVKey = 'alphaKey';
def getSandP500():
    resp = requests.get('https://www.slickcharts.com/sp500');
    soup = bs.BeautifulSoup(resp.text, 'lxml');
    table = soup.find('table',{'class': 'table-hover'});
    symbols=[]
    for row in table.findAll('tr')[1:]:
        symbol = row.findAll('td')[2].text
        symbols.append(symbol)
    return symbols;

def getAllStocks():
    stocks = getSandP500();
    for stock in stocks:
        if not os.path.exists("historicDataAdjustedLive/"+stock+".csv"):
            getDataPerSymbol(stock);
    return stocks;
def selectStocks():
    stNum=50;
    dfRank = pd.DataFrame(columns=['symbol','returns','volatility']);
    dfRank.set_index('symbol',inplace=True);
    for file in os.listdir("historicDataAdjustedLive/"):
        df = pd.read_csv("historicDataAdjustedLive/"+file);
        pd.to_datetime(df['date']);
        df.set_index('date',inplace=True);
        df['returns'] = (df['5. adjusted close'] - df['5. adjusted close'].shift(-1))*100;
        df = df.iloc[::-1]
        df = df.replace(np.nan,0.0)
        df['movStdDev']=df['returns'].rolling(min_periods=40,window=40).std()
        df.dropna(inplace=True)
        df = df[:50].copy();
        dfRank.loc[file[:-4]]=0;
        dfRank.at[file[:-4],'returns']=df['returns'].sum();
        dfRank.at[file[:-4],'volatility']=df.iloc[0]['movStdDev'];
    dfRank['totalRank']=dfRank['returns']*(1/dfRank['volatility']);
    accionesAEntrenar = dfRank.sort_values(by=["totalRank"],ascending=False).tail(stNum)
    return accionesAEntrenar.index;
def getDataPerSymbol(symbol):
    ts = TimeSeries(key=alphaVKey, output_format='pandas',indexing_type='date');

    df, meta_data = ts.get_daily_adjusted(symbol=symbol,
                         outputsize='full');
    df.to_csv("historicDataAdjustedLive/"+symbol+".csv")
    time.sleep(10)
def createTradeDfs(symbol):

    getDataPerSymbol(symbol)
    dfIni = pd.read_csv("historicDataAdjustedLive/"+symbol+'.csv');

    dfTraining = createDataFrameLstm(dfIni);
    dfTraining['symbol']=symbol
    
    return dfTraining
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def createDataFrameHistories(histories):
    d = {};
    for h in histories:
        p=moving_average(histories[h]['val_f0'],1)
        q=moving_average(histories[h]['val_f1'],1)
        r=moving_average(histories[h]['val_f2'],1)
        pe=moving_average(histories[h]['f0'],1)
        qd=moving_average(histories[h]['f1'],1)
        rd=moving_average(histories[h]['f2'],1)
        d[h]={};
        d[h]['Clase 0']=p[-1]
        d[h]['Clase 1']=q[-1]
        d[h]['Clase 2']=r[-1]
        d[h]['Clase 0 tr']=pe[-1]
        d[h]['Clase 1 tr']=qd[-1]
        d[h]['Clase 2 tr']=rd[-1]
    return d;

def createDataFrameLstm(df):
    df = df.copy();
    df =getRankingNotNorm(df);
 
    df = deleteNans(df);
    exp1 = df['5. adjusted close'].ewm(span=12, adjust=False).mean()
    exp2 = df['5. adjusted close'].ewm(span=26, adjust=False).mean()
    df['ewm12']=exp1;
    df['ewm26']=exp2;
    df['ewm40']=df['5. adjusted close'].ewm(span=40, adjust=False).mean()
    macd = exp1 - exp2
    df['macd']=macd;
    import math
    middleBand = df['5. adjusted close'].rolling(min_periods=1,window=20).mean();
    topBand = middleBand + 2 * df['5. adjusted close'].rolling(min_periods=1,window=20).std();
    bottomBand = middleBand - 2 * df['5. adjusted close'].rolling(min_periods=1,window=20).std();
    diffBand = (topBand - bottomBand)/middleBand;
    df['topband']= topBand;
    df['priceEwm']=df['5. adjusted close']-df['ewm12']
    df['ma20'] = middleBand;
    df['lowerBand'] = bottomBand;
    df['bollWidth'] = diffBand;
    df['logBollWidth'] = np.log(diffBand.astype('float64'));
    df['StdRetdelta10']=df['movStdDev']-df.shift(10)['movStdDev'];
    df['StdRetdelta20']=df['movStdDev']-df.shift(20)['movStdDev'];
    df['mmi100'] = MMInp(df['5. adjusted close'].values)
    df['mmi50'] = MMInp(df['5. adjusted close'].values,period=50)
    df['diffToMeanVol30']=df['6. volume']-df['6. volume'].rolling(window=30,min_periods=0).mean();
    df['diffToMeanVol10']=df['6. volume']-df['6. volume'].rolling(window=10,min_periods=0).mean();
    df['RSI25']=RSI(df,25)/100
    df['RSI15']=RSI(df,15)/100
    df['MMI50delta10']=df['mmi50']-df.shift(10)['mmi50']
    df['bollWidth10']=diffBollWidth(df,10);
    df['bollWidth20']=diffBollWidth(df,20);
    df['hurst'] = HurstExponent(df)
    df['RSI25Log']=(df['RSI25']-df['RSI25'].min())/(df['RSI25'].max()-df['RSI25'].min())
    #df = df.replace(0,0.001);
    df['RSI25Log']=np.log(df['RSI25Log']);
    df['MMI50delta10'].clip(lower=df['MMI50delta10'].quantile(q=0.95))
    target(df,'target4_8');
    #targetAntiguo(df,4,2.5,8,'target4_8')
    
    return df;


from numpy import cumsum, log, polyfit, sqrt, std, subtract
windowOfTarget=50;
def rankingStocks():
    dfRank = pd.DataFrame(columns=['symbol','returns','volatility']);
    dfRank.set_index('symbol',inplace=True);
    for file in os.listdir("historicDataAdjNorm"):
        df=pd.read_csv("historicDataAdjNorm/"+file);
        df.set_index('date',inplace=True);
        df.dropna(inplace=True)
        df = df.iloc[::-1]
        df = df[:windowOfTarget].copy();
        #dfNew = pd.DataFrame([file[:-4]],[df['returns'].sum()],[df.iloc[0]['movStdDev']]);
        dfRank.loc[file[:-4]]=0;
        dfRank.at[file[:-4],'returns']=df['returns'].sum();
        dfRank.at[file[:-4],'volatility']=df.iloc[0]['movStdDev'];
    return dfRank;
def deleteNans(df):
    df = df.replace(np.nan,0.00)
    return df;
def zscore(x, window):
    x = x.iloc[::-1]
    r = x.rolling(window=window,min_periods=1)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z
def getOutliersPerFile(file):
    df = pd.read_csv(file);
    pd.to_datetime(df['date']);
    df.set_index('date',inplace=True);
    df.dropna(inplace=True);
    df = df[['5. adjusted close','returns']]
    z = np.where(zscore(df,10) > 3);
    if len(z)>2:
        print(file)
def MMInp(data, period = 100) :
    nLen = len(data);
    mmi = np.zeros(nLen);
    for i in range(len(data)):
        periodx = min(nLen-i,period)
        dataToSee = data[nLen-1-i-periodx:nLen-1-i]
        m = np.median(dataToSee);
        nRev = 0;
        for j in range(len(dataToSee)-1):
            prev = dataToSee[j]
            post = dataToSee[j+1];
            if post>m and post>prev:
                nRev+=1;
            elif post<m and post<prev:
                nRev+=1;
        mmi[nLen-i-1]=100.*(nRev/periodx);
    return mmi;


def RSI(df,n):
    delta = df['returns']
    pos= delta * 0;
    neg= delta * 0;
    i_pos = delta > 0
    i_neg = delta <= 0
    pos[i_pos] = delta[i_pos]
    neg[i_neg] = abs(delta[i_neg]);
    rs = pos.ewm(span=n).mean() / neg.ewm(span=n).mean()
    return 100 - 100 / (1 + rs)
def RSILast(df,n):
    delta = df['returns']
    pos= delta * 0;
    neg= delta * 0;
    i_pos = delta > 0
    i_neg = delta <= 0
    pos[i_pos] = delta[i_pos]
    neg[i_neg] = abs(delta[i_neg]);
    rs = pos.ewm(span=n).mean() / neg.ewm(span=n).mean()
    return 100 - 100 / (1 + rs)
def MMInpLast(data, period = 100) :
    nLen = len(data);
    mmi = np.zeros(nLen);
    i = 0
    periodx = min(nLen-i,period)
    dataToSee = data[nLen-1-i-periodx:nLen-1-i]
    m = np.median(dataToSee);
    nRev = 0;
    for j in range(len(dataToSee)-1):
        prev = dataToSee[j]
        post = dataToSee[j+1];
        if post>m and post>prev:
            nRev+=1;
        elif post<m and post<prev:
            nRev+=1;
    return 100.*(nRev/periodx);

def diffBollWidth(df,x):
    df = df.copy()
    df['width'] = np.exp(df['logBollWidth'])
    return df['width']-df.shift(x)['width'];
def HurstExponent(df):
    data = df['5. adjusted close'].values;
    period = 1000;
    nLen = len(data);
    hurst = np.zeros(len(data));
    for i in range(len(data)-20):
        startN = max(0,nLen-1000-i)
        ts = data[startN:nLen-i];
        lags = range(2, 20)
        tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        m = polyfit(log(lags), log(tau), 1)
        hurst[nLen-i-1]=m[0]*2.0
    return hurst;   
def HurstExponentLast(df):
    data = df['5. adjusted close'].values;
    period = 1000;
    nLen = len(data);
    hurst = np.zeros(len(data));
    i=0
    startN = max(0,nLen-1000-i)
    ts = data[startN:nLen-i];
    lags = range(2, 20)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    m = polyfit(log(lags), log(tau), 1)
    return m[0]*2.0
def targetAntiguo(df,profitRate,stopLossRate,period,column):
    df[column]=1;
    df['topWin']=0.0;#sirven como marca cuando se realize la estrategia
    df['lowWin']=0.0;
    for index, row in df.iterrows():
        i = df.index.get_loc(index)
        rows = df.iloc[i:i+period];
        startPrice = row['5. adjusted close'];
        targetHighPrice = startPrice + row['movStdDev']/100 * profitRate;
        targetLowPrice = startPrice - row['movStdDev']/100 * profitRate;
        df.loc[index,'topWin']=targetHighPrice;
        df.loc[index,'lowWin']=targetLowPrice;
        stopLossHigh = startPrice - row['movStdDev']/100 * stopLossRate;
        stopLossLow = startPrice + row['movStdDev']/100 * stopLossRate;
        stopLossHighB = False;
        stopLossLowB = False;
        price = startPrice;
        for index2,row2 in rows.iterrows():
            price = row2['5. adjusted close'];
            if price <=  stopLossHigh :
                stopLossHighB = True;
            if price >= stopLossLow :
                stopLossLowB= True;
            if price >= targetHighPrice and not stopLossHighB : 
                df.loc[index,column] = 2;
            if price <= targetLowPrice and not stopLossLowB : 
                df.loc[index,column] = 0;
def target(df,column):

    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2
    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
              size = total-(window_size)+1
    """
    window_size=11;
    col_name='5. adjusted close';
    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = (window_begin + window_end) / 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[row_counter] = 0
            elif min_index == window_middle:
                labels[row_counter] = 2
            else:
                labels[row_counter] = 1

        row_counter = row_counter + 1
    df['target4_8']=labels;
    return labels
    
def getRankingNotNorm(df):
    #df['date']=pd.to_datetime(df['date']);
    df.set_index('date',inplace=True);
    df['returns'] = (df['5. adjusted close'] - df['5. adjusted close'].shift(-1))*100;
    df = df.iloc[::-1]
    df = df.replace(np.nan,0.0)
    df['movStdDev']=df['returns'].rolling(min_periods=0,window=40).std()        
    return df;
def getLstm(df):
    dfNew = pd.DataFrame();
    dfNew.date2='';
    val = [[]]
    dfD = df;
    dfD['date2']=dfD.index;
    window= 20
    for index,row in dfD.iterrows():
        i = dfD.index.get_loc(index)        
        if i-window>=0:
            rows = df.iloc[i-window+1:i+1];
            rows['date']=rows.index;
            rows['date2'] = index;
            rows['date']=pd.to_datetime(rows['date']);
            rows['date2']=pd.to_datetime(rows['date2']);
            indexmi=rows[['symbol','date2','date']];
            index  =pd.MultiIndex.from_frame(indexmi,names=['symbol','date2','date']);
            rows.set_index(index,inplace=True);
            if dfNew.empty:
                dfNew = rows;
            else:
                dfNew = dfNew.append(rows)

    dfNew['date']=dfNew.index
    return dfNew;
def normalize(df):
    df = df.copy();
    df = df.iloc[15:];
    columns = df.columns[8:-6]
    for column in columns:     
        df[column].replace(np.nan,0.0,inplace=True)
        if column != 'movStdDev' and column!='target4_8':
            if column != 'macd':
                df[column] = (df[column]-df[column].rolling(50,min_periods=1).mean())/(df[column].rolling(window=50,min_periods=1).std());
            if column =='macd':
                df[column] = (df[column]-0.0)/(df[column].rolling(window=50,min_periods=1).std());
        df[column].replace(np.nan,0.0,inplace=True)
        #cambioamos los negative NINF para evitar los problemas del MMI50 y 100 debido alla fucnión del quartile
        df.replace(np.NINF,0.0,inplace=True);
        df.replace(np.inf,0.0,inplace=True)

    return df;
