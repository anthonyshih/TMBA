#### 進場 ＭＡＣＤ
#### 出場 ＭＡＣＤ＋ＲＳＩ

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import requests
import datetime as dt
import time
import quantstats
import os


import plotly
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
pyo.init_notebook_mode()

import plotly.io as io


plt.style.use('ggplot')
pd.set_option('display.max_rows', None)

## 獨台指期 1 分K
df= pd.read_csv('TWF_Futures_Minute_Trade.txt')


## 整理資料
df.index = pd.to_datetime(df['Date'] + ' ' + df['Time']) ## 中間要加空格 pandas 的用法
df = df.drop(columns=['Date','Time'])
df.columns = ['open', 'high', 'low', 'close', 'volume']
df['Hour'] = df.index.map(lambda x: x.hour)


rule = '3T' ## 3T 代表3個時間單位，在此就是3分鐘
Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)]
Morning.index = Morning.index + dt.timedelta(minutes=15) #先加 15 分鐘 以 9:00 單位開始

# 確認 9:03 及 9:06 的 open 是 9:01 及 9:04 的 open
Morning.resample(rule=rule, closed='right', label='right').first()[['open']].iloc[0:3]

# label='left' => 時間轉為左側時間
Morning.resample(rule=rule, closed='right', label='left').first()[['open']].iloc[0:3]

## 只抓日盤 =>時間只在8~13點之間
rule = '60T' ##以1個小時為單位

Morning = df[(df['Hour'] >= 8) & (df['Hour'] <= 13)]
Morning.index = Morning.index + dt.timedelta(minutes=15)

d1 = Morning.resample(rule=rule, closed='right', label='left').first()[['open']] #只要 open 的第一個值
d2 = Morning.resample(rule=rule, closed='right', label='left').max()[['high']]
d3 = Morning.resample(rule=rule, closed='right', label='left').min()[['low']]
d4 = Morning.resample(rule=rule, closed='right', label='left').last()[['close']]
d5 = Morning.resample(rule=rule, closed='right', label='left').sum()[['volume']]

df_Morning = pd.concat([d1,d2,d3,d4,d5], axis=1)
df_Morning = df_Morning.dropna()
df_Morning.index = df_Morning.index - dt.timedelta(minutes=15)

## 只抓夜盤 =>時間不能是8~13點之間
rule = '60T'

Night = df[(df['Hour'] < 8) | (df['Hour'] > 13)]

d1 = Night.resample(rule=rule, closed='right', label='left').first()[['open']]
d2 = Night.resample(rule=rule, closed='right', label='left').max()[['high']]
d3 = Night.resample(rule=rule, closed='right', label='left').min()[['low']]
d4 = Night.resample(rule=rule, closed='right', label='left').last()[['close']]
d5 = Night.resample(rule=rule, closed='right', label='left').sum()[['volume']]

df_Night = pd.concat([d1,d2,d3,d4,d5], axis=1)
df_Night = df_Night.dropna()

## 完整資料: 日盤 + 夜盤
df_Day = pd.concat([df_Morning, df_Night], axis=0)
df_Day = df_Day.sort_index(ascending=True)



## 選擇回測資料
# df_Morning['Month'] = df_Morning.index.map(lambda x: x.month)
# df_Morning['Weekday'] = df_Morning.index.map(lambda x: x.weekday)+1
df_Morning['Hour'] = df_Morning.index.map(lambda x: x.hour)


#將資料切成兩份
trainData = df_Morning[(df_Morning.index >= '2011-01-01 00:00:00') & (df_Morning.index <= '2019-12-31 00:00:00')].copy()  # trend data
testData = df_Morning[(df_Morning.index >= '2020-1-1 00:00:00') & (df_Morning.index <= '2022-5-22 00:00:00')].copy()      # test data


## 取得結算日期
settlementDate_ = pd.read_csv('settlementDate.csv')#, encoding = 'ANSI')
settlementDate_.columns = ['settlementDate', 'futures', 'settlementPrice']

#只抓取資料沒有 "W" 的(因為 W 代表周結算)
bool_ = [False if 'W' in i else True for i in settlementDate_['futures']]
settlementDate = [i.replace('/','-') for i in list(settlementDate_[bool_]['settlementDate'])]
settlementDate = [pd.to_datetime(i).date() for i in settlementDate]




# 策略參數最佳化
optimizationList = []
fund = 1000000
feePaid = 600
length_1 = 20
length_2 = 60
length_3 = 9
periods = 14



# RSI函數
def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

# 最佳化參數：length & NumStd
for length_1 in range(15,40,5): 
    for length_2 in range(15,55,5) :
        for length_3 in range(3,27,3):
            for periods in range(16,24,2):
                if length_2 > length_1:
                    print('----------')
                    print(f'length_1: {length_1}')
                    print(f'length_2: {length_2}')
                    print(f'length_3: {length_3}')
                    print(f'periods: {periods}')
                    
                    trainData['EMA_1'] = trainData['close'].ewm(span=length_1, adjust=False).mean()
                    trainData['EMA_2'] = trainData['close'].ewm(span=length_2, adjust=False).mean()
                    trainData['MACD'] = trainData['EMA_1'] - trainData['EMA_2']
                    trainData['Signal'] = trainData['MACD'].ewm(span=length_3, adjust=False).mean()
                    trainData['RSI']=rsi(trainData, periods , ema = True)

                            
                    #將時間序列數據轉換為NumPy數組
                    df_arr = np.array(trainData)
                    time_arr = np.array(trainData.index)
                    date_arr = [pd.to_datetime(i).date() for i in time_arr]

        ##  做多

                    BS = None
                    buy = []
                    sell = []
                    profit_list = [0]
                    profit_fee_list = [0]
                    profit_fee_list_realized = []
                    rets = []

                    for i in range(len(df_arr)):

                        if i == len(df_arr)-1:
                            break

                        ## 進場邏輯
                        entryLong = df_arr[i,8] > df_arr[i,9] 
                        entryCondition = date_arr[i] not in settlementDate

                        ## 出場邏輯
                        exitShort = df_arr[i,8] < df_arr[i,9] and df_arr[i,10] < 50                        
                        exitCondition = date_arr[i] in settlementDate and df_arr[i,5] >= 11

                    #做多的狀態
                        if BS == 'B':
                        # 停損條件
                        #以達成進場條件當作關鍵 K 棒，並設其最低點為停損點
                            stopLoss  = df_arr[i,3] <= df_arr[t,2] 


                        if BS == None:
                            profit_list.append(0)
                            profit_fee_list.append(0)

                            if entryLong and entryCondition:
                                BS = 'B'
                                t = i+1
                                buy.append(t)                            

                        elif BS == 'B':
                            profit = 200 * (df_arr[i+1,0] - df_arr[i,0])
                            profit_list.append(profit) #紀錄損益

                            if exitShort or i == len(df_arr)-2 or exitCondition or stopLoss :
                                pl_round = 200 * (df_arr[i+1,0] - df_arr[t,0])
                                profit_fee = profit - feePaid*2
                                profit_fee_list.append(profit_fee)
                                
                                sell.append(i+1)
                                BS=None

                                # Realized PnL
                                profit_fee_realized = pl_round - feePaid*2
                                profit_fee_list_realized.append(profit_fee_realized)
                                rets.append(profit_fee_realized/(200*df_arr[t,0]))

                            else:
                                profit_fee = profit
                                profit_fee_list.append(profit_fee)

                    equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=trainData.index)
                    equity['equity'] = equity['profitfee'] + fund
                    equity['drawdown_percent'] = (equity['equity'] / equity['equity'].cummax()) - 1
                    equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
                    ret = equity['equity'][-1]/equity['equity'][0] - 1
                    mdd = abs(equity['drawdown_percent'].min())
                    calmarRatio = ret / mdd

                    optimizationList.append([length_1, length_2, length_3, periods, ret, calmarRatio])

#print(optimizationList)

optResult = pd.DataFrame(optimizationList, columns=['length_1', 'length_2', 'length_3','periods','ret', 'calmarRatio'])
#print(optResult.head())



max_ret_index = optResult['ret'].idxmax()
max_ret_values = optResult.loc[max_ret_index]
#print(max_ret_values)

max_ret = optResult['ret'].max()
#print(max_ret)

max_ret_values = optResult.loc[max_ret_index, ['length_1', 'length_2', 'length_3','periods','ret', 'calmarRatio']]
print(max_ret_values)



