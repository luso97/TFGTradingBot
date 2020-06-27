'''
Created on 20 may. 2020

@author: Luis
'''
import tensorflow as tf
import numpy.random as rd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Dense, Dropout, LSTM,  BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
import time
import pickle
import alpaca_trade_api as tradeapi
import os
from sklearn.utils import class_weight
from datetime import date
import pandas as pd
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract
def f1(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc

def f2(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 2), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc
def f0(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 0), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc

apiKey = 'apiKey'
apiSecret ='apisecret'
alphaVKey='z'

class Estrategia():
    def __init__(self,instruments, dfTrade,window,profit,stopLoss,periodo,startT):
        self.__lposition = None
        self.instruments = instruments;
        self.api = tradeapi.REST(
        apiKey,
        apiSecret,
        'https://paper-api.alpaca.markets')
        self.__sposition = None
        self.profit = profit;
        self.stopLoss = stopLoss;
        self.periodo = periodo;
        self.window=window;
        self.predictions={};
        self.investingL={};
        self.investingS={};
        self.probE={}
        
        self.waiting={};
        self.startT = startT;
        self.next={};
        self.periodoL ={};
        self.periodoS={};
        self.df = dfTrade;
        for s in self.instruments:
            self.probE[s]=0.1
            self.predictions[s]=[];
            self.next[s]=1
            self.periodoL[s]=0
            self.periodoS[s]=0
            self.waiting[s]=0;
        self.columns=self.df[self.instruments[0]].columns[8:-6];
        self.model={};
        for ins in self.instruments:
            if os.path.exists("model"+ins+".h5"):
                self.model[ins] = tf.keras.models.load_model("model"+ins+".h5",custom_objects ={'f1':f1,'f2':f2,'f0':f0})
            else:
                self.model[ins] = self.train_model(ins);
                self.model[ins].save('model'+ins+'.h5');
    def entrarLong(self,cant,sym):
        print("Posicion larga con "+str(cant)+" acciones de "+sym);
        try:
            self.api.submit_order(
                symbol=sym,
                qty=cant,
                side='buy',
                type='market',
                time_in_force='day'
            )
        except:
            print("Error creando posicion");
        
        account = self.api.get_account()
        print("Nuestro equity es: "+account.equity) 
    def entrarShort(self,cant,sym):
        print("Posicion corta con "+str(cant)+" acciones de "+sym);
        try:
            self.api.submit_order(
                symbol=sym,
                qty=int(cant),
                side='sell',
                type='market',
                time_in_force='day'
            )
        except:
            print("Error creando posicion")
        
        account = self.api.get_account()
        print("Nuestro equity es: "+account.equity) 
    def salirDePosicion(self,sym,position):
        s='';
        cant = int(position.qty);
        print("Salida de posicion de "+sym+", Posicion "+position.side);
        if(position.side=='short'):
            s = 'buy'
            cant = -cant
        else:
            s = 'sell'
        account = self.api.get_account()
        print("Nuestro equity es: "+account.equity) 
        try:
            self.api.submit_order(
                symbol=sym,
                qty=cant,
                side=s,
                type='market',
                time_in_force='day'
            )
        except:
            print("Error saliendo de posicion, puede que orden este en proceso")
    def getPositions(self):
        return self.api.list_positions()
    def getStatus(self):
        if(self.api.get_clock().is_open):
            positions = self.getPositions()
            for pos in positions:
                if float(pos.unrealized_plpc) > self.profit or 1==1:
                    self.salirDePosicion(pos.symbol,pos)
                elif float(pos.unrealized_plpc) < -self.stopLoss:
                    self.salirDePosicion(pos.symbol,pos)
    def addFilas(self):
        if(self.api.get_clock().is_open):
            calendar = self.api.get_calendar(start='2020-01-10', end=date.today())[-2:]
            for ins in self.instruments:
                barset = self.api.get_barset(ins, '1D',limit=1)
                bar = barset[ins][0]
                print(calendar)
                self.addRow(ins,bar.c,calendar[0].date,bar.v)
    def predict(self,sym):
        dfTradeNorm = self.normalize(self.df[sym])
        dfCols = dfTradeNorm[self.columns]

        #if(np.cumsum(d3)[-1])>0.2:
        #    print(d3);
        #    print(self.dfTry.loc['ROL',date.strftime("%Y-%m-%d"),date.strftime("%Y-%m-%d")][self.columns])
        #    print(dfCols.loc[date])
        #print(self.window)
        window=self.window
        res = dfCols.iloc[-window:].values;
        #print(res.shape)
        result = self.model[sym].predict([res.tolist()])[0];
        return result;
    def signalOrders(self):
        if(self.api.get_clock().is_open):
            ins = [] #acciones ya con posiciones activas
            positions = self.getPositions();
            for pos in positions:
                ins.append(pos.symbol)
            
            for sym in self.instruments:

                ed=self.predict(sym)
                e=np.argmax(ed);
                self.predictions[sym].append((e)*40);
                e=2;
                # Wait for enough bars to be available to calculate a SMA.
                if e==0 or e==2:
                    if self.waiting[sym]==0:
                        self.next[sym]=e;
                        self.probE[sym] =ed[e];
                    self.waiting[sym]=1;
            
            for sym in self.instruments:
                if sym not in ins:

                    if self.next[sym]==2:
                        if self.waiting[sym]==0:
                            self.next[sym]=2;
                        self.periodoL[sym] += 1 ;
                        if self.periodoL[sym] == self.startT:
                            account = self.api.get_account()
                            barset = self.api.get_barset(sym, '1Min',limit=1)
                            bar = barset[sym][0]
                            x=int((float(account.cash) /bar.c)*(min(self.probE[sym],0.95)));
                            self.periodoL[sym]= 0;
                            self.investingL[sym]=0;
                            self.next[sym]=1;
                            self.waiting[sym]=0;
                            #self.info("LONG")
                            self.entrarLong(x,sym);

                    elif self.next[sym]==0:
                        if self.waiting[sym]==0:
                            self.next[sym]=0;
                        self.periodoS[sym] += 1;
                        if self.periodoS[sym]==self.startT or 1==1: 
                            account = self.api.get_account()
                            barset = self.api.get_barset(sym, '1Min',limit=1)
                            bar = barset[sym][0]
                            x=int((float(account.cash) /bar.c)*(min(self.probE[sym],0.95)));
                            self.periodoS[sym] = 0;
                            self.investingS[sym]=0;
                            #self.info("SHORT");
                            self.next[sym]=1
                            self.waiting[sym]=0;
                            self.entrarShort(x,sym);
        with open('dicts/periodoS.pickle', 'wb') as handle:
            pickle.dump(self.periodoS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/periodoL.pickle', 'wb') as handle:
            pickle.dump(self.periodoL, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/waiting.pickle', 'wb') as handle:
            pickle.dump(self.waiting, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/next.pickle', 'wb') as handle:
            pickle.dump(self.next, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/investingS.pickle', 'wb') as handle:
            pickle.dump(self.investingS, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/investingL.pickle', 'wb') as handle:
            pickle.dump(self.investingL, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/predictions.pickle', 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/probE.pickle', 'wb') as handle:
            pickle.dump(self.probE, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('dicts/instruments.pickle', 'wb') as handle:
            pickle.dump(self.instruments, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    def load(self):
        with open('dicts/next.pickle', 'rb') as handle:
            self.next = pickle.load(handle)
        with open('dicts/periodoS.pickle', 'rb') as handle:
            self.periodoS = pickle.load(handle)
        with open('dicts/periodoL.pickle', 'rb') as handle:
            self.periodoL = pickle.load(handle)
        with open('dicts/investingL.pickle', 'rb') as handle:
            self.investingL = pickle.load(handle)
        with open('dicts/investingS.pickle', 'rb') as handle:
            self.investingS = pickle.load(handle)
        with open('dicts/predictions.pickle', 'rb') as handle:
            self.predictions = pickle.load(handle)
        with open('dicts/probE.pickle', 'rb') as handle:
            self.probE = pickle.load(handle)
        with open('dicts/waiting.pickle', 'rb') as handle:
            self.waiting = pickle.load(handle)
        with open('dicts/instruments.pickle', 'rb') as handle:
            self.instruments = pickle.load(handle)

        dfs={};
        self.model={};
        for sym in self.instruments:
            df = pd.read_csv("strat/"+sym+".csv")
            df.set_index('date',inplace=True);
            dfs[sym]=df
            self.model[sym] = tf.keras.models.load_model("model"+sym+".h5",custom_objects ={'f1':f1,'f2':f2,'f0':f0})
        self.df=dfs;

        
    def addRow(self,sym,price,date,volume):
        dfTrad = self.df[sym];
        dfRow=pd.DataFrame({'date':date,'5. adjusted close':price,'6. volume':volume},index=[0]);
        dfRow.set_index('date',inplace=True);          
        dfTrad=dfTrad.append(dfRow,ignore_index=False);
        dfTrad.at[date,'returns']=(dfTrad.iloc[-1]['5. adjusted close']-dfTrad.iloc[-2]['5. adjusted close'])*100
        dfTrad.at[date,'movStdDev']=dfTrad.iloc[-400:]['returns'].rolling(min_periods=0,window=40).std()[-1];    
        exp1 = dfTrad['5. adjusted close'].ewm(span=12, adjust=False).mean()
        exp2 = dfTrad['5. adjusted close'].ewm(span=26, adjust=False).mean()
        dfTrad['ewm12']=exp1;
        dfTrad['ewm26']=exp2;
        dfTrad['ewm40']=dfTrad['5. adjusted close'].ewm(span=40, adjust=False).mean()
        macd = exp1[-1] - exp2[-1]
        dfTrad.at[date,'macd']=macd;
        dfTrad['ewm40']=dfTrad['5. adjusted close'].ewm(span=40,adjust=False).mean()
        dfTrad.tail()
        middleBand = dfTrad.iloc[-20:]['5. adjusted close'].rolling(min_periods=1,window=20).mean()[-1]
        topBand = middleBand + 2 * dfTrad.iloc[-20:]['5. adjusted close'].rolling(min_periods=1,window=20).std()[-1]
        bottomBand = middleBand - 2 * dfTrad.iloc[-20:]['5. adjusted close'].rolling(min_periods=1,window=20).std()[-1]
        diffBand = (topBand - bottomBand)/middleBand;
        dfTrad.at[date,'topband']= topBand;
        dfTrad['priceEwm']=price-exp1[-1];
        dfTrad.at[date,'ma20'] = middleBand;
        dfTrad.at[date,'lowerBand'] = bottomBand;
        dfTrad.at[date,'bollWidth'] = diffBand;
        dfTrad.at[date,'logBollWidth'] = np.log(diffBand.astype('float64'));
        dfTrad.at[date,'StdRetdelta10']=dfTrad['movStdDev'][-1]-dfTrad['movStdDev'][-11];
        dfTrad.at[date,'StdRetdelta20']=dfTrad['movStdDev'][-1]-dfTrad['movStdDev'][-21];
        dfTrad.at[date,'mmi100'] = self.MMInpLast(dfTrad.iloc[-130:]['5. adjusted close'].values)
        dfTrad.at[date,'mmi50'] = self.MMInpLast(dfTrad.iloc[-130:]['5. adjusted close'].values,period=50)
        dfTrad.at[date,'diffToMeanVol30']=dfTrad['6. volume'][-1]-dfTrad.iloc[-35:]['6. volume'].rolling(window=30,min_periods=0).mean()[-1];
        dfTrad.at[date,'diffToMeanVol10']=dfTrad['6. volume'][-1]-dfTrad.iloc[-35:]['6. volume'].rolling(window=10,min_periods=0).mean()[-1];
        dfTrad.at[date,'RSI25']=(self.RSI(dfTrad,25)/100) [-1]
        dfTrad.at[date,'RSI15']=(self.RSI(dfTrad,15)/100) [-1]
        dfTrad.at[date,'MMI50delta10']=dfTrad.iloc[-1]['mmi50']-dfTrad.iloc[-11]['mmi50']
        dfTrad.at[date,'bollWidth10']=np.exp(dfTrad.iloc[-1]['logBollWidth'])- np.exp(dfTrad.iloc[-11]['logBollWidth'])
        dfTrad.at[date,'bollWidth20']=np.exp(dfTrad.iloc[-1]['logBollWidth'])- np.exp(dfTrad.iloc[-21]['logBollWidth'])
        dfTrad.at[date,'hurst'] = self.HurstExponentLast(dfTrad);
        dfTrad['RSI25Log']=np.log((dfTrad['RSI25']-dfTrad['RSI25'].min())/(dfTrad['RSI25'].max()-dfTrad['RSI25'].min()))
        targetHighPrice = price + (dfTrad.loc[date,'movStdDev']/100) * 2;
        targetLowPrice = price - (dfTrad.loc[date,'movStdDev']/100) * 2;
        #print(dfTrad.loc[date,'movStdDev'],price, targetHighPrice);
        dfTrad.at[date,'topWin']=targetHighPrice;
        dfTrad.at[date,'lowWin']=targetLowPrice;
        stopLossHigh = price -  (dfTrad.loc[date,'movStdDev']/100) * 3;
        stopLossLow = price + (dfTrad.loc[date,'movStdDev']/100) * 3;
        dfTrad.at[date,'topLoss']=stopLossHigh;
        dfTrad.at[date,'lowLoss']=stopLossLow;
        dfTrad.at[date,'topLoss']=stopLossHigh;
        dfTrad.at[date,'lowLoss']=stopLossLow;
        
        self.df[sym] = dfTrad;

        dfTrad.to_csv("strat/"+sym+".csv")
    def HurstExponentLast(self,df):
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
    def MMInpLast(self,data, period = 100) :
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
    def RSI(self,df,n):
        delta = df['returns']
        pos= delta * 0;
        neg= delta * 0;
        i_pos = delta > 0
        i_neg = delta <= 0
        pos[i_pos] = delta[i_pos]
        neg[i_neg] = abs(delta[i_neg]);
        rs = pos.ewm(span=n).mean() / neg.ewm(span=n).mean()
        return 100 - 100 / (1 + rs)
    def train_model(self,sym):
        print('training of dataset started')
        #normalization
        df=self.df[sym];
        columns=df.columns[8:-6]
        dfTrain =self.normalize(df)

        #Creación dataset LSTM 
        dfLstmTrain = self.getLstm(dfTrain);

        #Filtramos las columnas
        dfColsTrain = dfLstmTrain[self.columns]
        res = [];
        y=[]
        rr=True

        for val in dfColsTrain.index.get_level_values(0).unique():
            for val2 in dfColsTrain.loc[val].index.get_level_values(0).unique():
                res.append(dfColsTrain.loc[val,val2].values);
                y.append(dfLstmTrain.loc[val,val2,val2]['target4_8'])
        resnp = np.asarray(res)
        ynp = np.asarray(y)

        #resnp=np.delete(resnp,slice(0,10),1)
        from sklearn.utils import shuffle
        X_total,y_total = shuffle(resnp,ynp);

        #X_total=tf.keras.utils.normalize(X_total,axis=1,order=1)
        x_train=X_total;
        y_train=tf.keras.utils.to_categorical(
            y_total, num_classes=3, dtype='int16'
        )
        x_test=X_total[-100:];
        y_test=tf.keras.utils.to_categorical(
            y_total[-100:], num_classes=3, dtype='int16'
        )
        model2 = Sequential()


        model2.add(LSTM(128, input_shape=(x_train.shape[1:]),return_sequences=False))
        model2.add(Dropout(0.1))
        model2.add(BatchNormalization()) 
        model2.add(Dense(3, activation='softmax'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        EPOCHS = 70  # how many passes through our data
        BATCH_SIZE = 60
        # Compile model

        def f1(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc

        def f2(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, 2), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        def f0(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask = K.cast(K.equal(class_id_preds, 0), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc

        class_weights = class_weight.compute_class_weight('balanced',
                                                 [0,1,2],
                                                 y_total)


        import tensorflow.keras.backend as K
        from sklearn.metrics import classification_report



        model2.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy',f1,f0,f2]
        )

        class_w={0:class_weights[0]*(1/class_weights[1]),1:1.0,2:class_weights[2]*(1/class_weights[1])}
        #print(class_w)
        #class_w = {0:5.5,1:1.0,2:4.5}
        model2.summary();
        history = model2.fit(
            x_train,y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            class_weight=class_w,

            verbose=0
        )

        return model2
    def getLstm(self,df):
        dfNew = pd.DataFrame();
        dfNew.date2='';
        val = [[]]
        dfD = df;
        dfD['date2']=dfD.index;
        window= self.window
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
    def normalize(self,df):
        df = df.copy();
        df = df.iloc[15:];
        columns = self.columns
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

        