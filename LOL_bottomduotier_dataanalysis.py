#!/usr/bin/env python
# coding: utf-8

# # 과제1, 바텀듀오의 티어

# ## 라이브러리, 데이터 로드

# In[937]:


import requests
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math


# In[4]:


url='*****************************************************'


# In[ ]:


adc_sup_pick_red


# In[ ]:


lol_data=data.text
lol_data=lol_data.replace('\n', ',\n')
lol_data='['+lol_data+']'
lol_data=lol_data.replace(']},\n]',']}\n]')


# In[ ]:


f = open("data.txt", 'w')
f.write(lol_data)
f.close()


# In[ ]:


lol_data=json.loads(lol_data)


# In[ ]:


output_df=json_normalize(lol_data)


# In[790]:


sample=output_df
sample.reset_index(inplace=True)
del sample['index']
del sample['Unnamed: 0']
sample


# ## 데이터 전처리

# ### teams
# #### 밴, 오브젝트에 대한 간략한 정보

# In[756]:


def array_on_duplicate_keys(ordered_pairs):
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            if type(d[k]) is list:
                d[k].append(v)
            else:
                d[k] = [d[k],v]
        else:
           d[k] = v
    return d


# In[757]:


teams_output = pd.DataFrame(columns = ['firstdragon', 'firstinhibitor', 'pickturn', 'championid', 'baronkills',
       'firstriftherald', 'firstbaron', 'riftheraldkills', 'firstblood',
       'teamid', 'firsttower', 'vilemawkills', 'inhibitorkills', 'towerkills',
       'dominionvictoryscore', 'win', 'dragonkills'])
def split_list(a_list):
    half = len(a_list)//2
    return a_list[:][:half], a_list[:][half:]


# In[758]:


for i in range(len(sample)):
    test=sample['teams'][i]
    test=test.replace("'", "\"").replace('[{','').replace('}]','').replace('}, {', ', ').replace(' "bans":','').replace('False','\"False\"').replace('True','\"True\"')
    test='[{' + test+ '}]'
    test=json.loads(test, object_pairs_hook=array_on_duplicate_keys)
    test=json_normalize(test)
    
    teams_output=pd.concat([teams_output,test])
    
teams_output.reset_index(inplace=True)
del teams_output['index']
teams_output.head()


# In[759]:


a=[]
b=[]

teams_output_blue=pd.DataFrame()
teams_output_red=pd.DataFrame()


for i in range(teams_output.shape[0]):
    for j in range(teams_output.shape[1]):
    
        A,B=split_list(teams_output.iloc[i][j])

        a.append(A)
        b.append(B)
    
    teams_output_blue=pd.concat([teams_output_blue,pd.DataFrame(pd.Series(a)).transpose()])
    teams_output_red=pd.concat([teams_output_red,pd.DataFrame(pd.Series(b)).transpose()])
    a=[]
    b=[]
    
teams_output_blue.columns=teams_output.columns
teams_output_red.columns=teams_output.columns

teams_output_blue.reset_index(inplace=True)
teams_output_red.reset_index(inplace=True)

teams_output_blue=teams_output_blue.rename({'championid':'championid_ban'},axis='columns')
teams_output_red=teams_output_blue.rename({'championid':'championid_ban'},axis='columns')

del teams_output_blue['index']
del teams_output_red['index']


# In[760]:


teams_output_blue.head()


# ### participants
# #### 팀 챔피언, 오브젝트, 킬에 대한 상세한 정보

# In[761]:


participants_output=pd.DataFrame()

for i in range(len(sample)):
    test=sample['participants'][i]
    test=test.replace("'", "\"").replace('[{','').replace('}]','').replace('}, {', ', ').replace(' "bans":','').replace('False','\"False\"').replace('True','\"True\"')
    test='[{' + test+ '}]'
    test=json.loads(test, object_pairs_hook=array_on_duplicate_keys)
    test=json_normalize(test)
    
    participants_output=pd.concat([participants_output,test])
participants_output.reset_index(inplace=True)
del participants_output['index']
participants_output.head()


# In[762]:


participants_output_if=pd.DataFrame(columns=['championid', 'kills', 'deaths', 'assists'])

for i in range(len(participants_output)):
    participants_output_if = participants_output_if.append(pd.DataFrame([[participants_output['championid'][i],
                                                                          list(json_normalize(participants_output['stats'][i])['kills']),
                                                                          list(json_normalize(participants_output['stats'][i])['deaths']),
                                                                          list(json_normalize(participants_output['stats'][i])['assists'])]], columns=['championid', 'kills', 'deaths', 'assists']), ignore_index=True)


# In[763]:


a=[]
b=[]

participants_output_if_blue=pd.DataFrame()
participants_output_if_red=pd.DataFrame()

for i in range(participants_output_if.shape[0]):
    for j in range(participants_output_if.shape[1]):
    
        A,B=split_list(participants_output_if.iloc[i][j])

        a.append(A)
        b.append(B)
    
    participants_output_if_blue=pd.concat([participants_output_if_blue,pd.DataFrame(pd.Series(a)).transpose()])
    participants_output_if_red=pd.concat([participants_output_if_red,pd.DataFrame(pd.Series(b)).transpose()])
    a=[]
    b=[]
    
participants_output_if_blue.columns=participants_output_if.columns
participants_output_if_red.columns=participants_output_if.columns

participants_output_if_blue.reset_index(inplace=True)
participants_output_if_red.reset_index(inplace=True)

del participants_output_if_blue['index']
del participants_output_if_red['index']


# In[764]:


participants_output_if_blue.head()


# ### gameduration
# #### 게임 시간

# In[765]:


sample['gameduration'].head()


# ### participantextendedstats
# #### 게임 플레이어들의 티어

# In[766]:


participantextendedstats_output=pd.DataFrame()

for i in range(len(sample)):
    test=sample['participantextendedstats'][i]
    test=test.replace("'", "\"").replace('[{','').replace('}]','').replace('}, {', ', ').replace(' "bans":','').replace('False','\"False\"').replace('True','\"True\"')
    test='[{' + test+ '}]'
    test=json.loads(test, object_pairs_hook=array_on_duplicate_keys)
    test=json_normalize(test)
    
    participantextendedstats_output=pd.concat([participantextendedstats_output,test])


# In[767]:


a=[]
b=[]

participantextendedstats_output_blue=pd.DataFrame()
participantextendedstats_output_red=pd.DataFrame()


for i in range(participantextendedstats_output.shape[0]):
    for j in range(participantextendedstats_output.shape[1]):
    
        A,B=split_list(participantextendedstats_output.iloc[i][j])

        a.append(A)
        b.append(B)
    
    participantextendedstats_output_blue=pd.concat([participantextendedstats_output_blue,pd.DataFrame(pd.Series(a)).transpose()])
    participantextendedstats_output_red=pd.concat([participantextendedstats_output_red,pd.DataFrame(pd.Series(b)).transpose()])
    a=[]
    b=[]
    
participantextendedstats_output_blue.columns=participantextendedstats_output.columns
participantextendedstats_output_red.columns=participantextendedstats_output.columns

participantextendedstats_output_blue.reset_index(inplace=True)
participantextendedstats_output_red.reset_index(inplace=True)

del participantextendedstats_output_blue['index']
del participantextendedstats_output_red['index']


# In[768]:


participantextendedstats_output_blue.head()


# ### champion info
# #### 챔피언들의 코드, 영문명, 한글명

# In[769]:


api_key = '**************************************************************'
r = requests.get('https://ddragon.leagueoflegends.com/api/versions.json') # version data 확인
current_version = r.json()[0] # 가장 최신 버전 확인
current_version


# In[770]:


r = requests.get('http://ddragon.leagueoflegends.com/cdn/{}/data/ko_KR/champion.json'.format(current_version))
parsed_data = r.json() 
ch_if = pd.DataFrame(parsed_data)


# In[771]:


ch_if_df=pd.DataFrame(columns=['key','name','id'])
for i in range(len(ch_if)):
    temp_df=ch_if['data'][i]
    temp_df=json_normalize(temp_df)[['key','name','id']]
    
    ch_if_df=pd.concat([ch_if_df,temp_df])

ch_if_df.reset_index(inplace=True)
del ch_if_df['index']


# In[772]:


ch_if_df


# ### 픽률 계산

# #### BLUE TEAM

# In[773]:


adc_sup_pick_blue=pd.DataFrame(participants_output_if_blue['championid'])
adc_sup_pick_blue['adc_champ']=""
adc_sup_pick_blue['adc_champ_name']=""
adc_sup_pick_blue['adc_champ_kill']=''
adc_sup_pick_blue['adc_champ_deaths']=''
adc_sup_pick_blue['adc_champ_assists']=''
adc_sup_pick_blue['sup_champ']=""
adc_sup_pick_blue['sup_champ_name']=""
adc_sup_pick_blue['sup_champ_kill']=''
adc_sup_pick_blue['sup_champ_deaths']=''
adc_sup_pick_blue['sup_champ_assists']=''
adc_sup_pick_blue['win']=teams_output_blue['win']


# In[774]:


def champ_name_korean(x): 
    if x!=-1: return ch_if_df.loc[ch_if_df['key'] == str(x), 'name'].values[0]
    return


# In[775]:


for i in range(len(adc_sup_pick_blue)):
    adc_sup_pick_blue['adc_champ'][i]=adc_sup_pick_blue['championid'][i][participantextendedstats_output_blue['position'][i].index('ADC')]
    adc_sup_pick_blue['adc_champ_name'][i]=champ_name_korean(adc_sup_pick_blue['adc_champ'][i])
    adc_sup_pick_blue['adc_champ_kill'][i]=participants_output_if_blue['kills'][i][participantextendedstats_output_blue['position'][i].index('ADC')]
    adc_sup_pick_blue['adc_champ_deaths'][i]=participants_output_if_blue['deaths'][i][participantextendedstats_output_blue['position'][i].index('ADC')]
    adc_sup_pick_blue['adc_champ_assists'][i]=participants_output_if_blue['assists'][i][participantextendedstats_output_blue['position'][i].index('ADC')]
    
    adc_sup_pick_blue['sup_champ'][i]=adc_sup_pick_blue['championid'][i][participantextendedstats_output_blue['position'][i].index('SUPPORT')]
    adc_sup_pick_blue['sup_champ_name'][i]=champ_name_korean(adc_sup_pick_blue['sup_champ'][i])
    adc_sup_pick_blue['sup_champ_kill'][i]=participants_output_if_blue['kills'][i][participantextendedstats_output_blue['position'][i].index('SUPPORT')]
    adc_sup_pick_blue['sup_champ_deaths'][i]=participants_output_if_blue['deaths'][i][participantextendedstats_output_blue['position'][i].index('SUPPORT')]
    adc_sup_pick_blue['sup_champ_assists'][i]=participants_output_if_blue['assists'][i][participantextendedstats_output_blue['position'][i].index('SUPPORT')]

del adc_sup_pick_blue['championid']


# #### RED TEAM

# In[776]:


adc_sup_pick_red=pd.DataFrame(participants_output_if_red['championid'])
adc_sup_pick_red['adc_champ']=""
adc_sup_pick_red['adc_champ_name']=""
adc_sup_pick_red['adc_champ_kill']=''
adc_sup_pick_red['adc_champ_deaths']=''
adc_sup_pick_red['adc_champ_assists']=''
adc_sup_pick_red['sup_champ']=""
adc_sup_pick_red['sup_champ_name']=""
adc_sup_pick_red['sup_champ_kill']=''
adc_sup_pick_red['sup_champ_deaths']=''
adc_sup_pick_red['sup_champ_assists']=''
adc_sup_pick_red['win']=teams_output_red['win']


# In[777]:


for i in range(len(adc_sup_pick_red)):
    adc_sup_pick_red['adc_champ'][i]=adc_sup_pick_red['championid'][i][participantextendedstats_output_red['position'][i].index('ADC')]
    adc_sup_pick_red['adc_champ_name'][i]=champ_name_korean(adc_sup_pick_red['adc_champ'][i])
    adc_sup_pick_red['adc_champ_kill'][i]=participants_output_if_red['kills'][i][participantextendedstats_output_red['position'][i].index('ADC')]
    adc_sup_pick_red['adc_champ_deaths'][i]=participants_output_if_red['deaths'][i][participantextendedstats_output_red['position'][i].index('ADC')]
    adc_sup_pick_red['adc_champ_assists'][i]=participants_output_if_red['assists'][i][participantextendedstats_output_red['position'][i].index('ADC')]
    
    adc_sup_pick_red['sup_champ'][i]=adc_sup_pick_red['championid'][i][participantextendedstats_output_red['position'][i].index('SUPPORT')]
    adc_sup_pick_red['sup_champ_name'][i]=champ_name_korean(adc_sup_pick_red['sup_champ'][i])
    adc_sup_pick_red['sup_champ_kill'][i]=participants_output_if_red['kills'][i][participantextendedstats_output_red['position'][i].index('SUPPORT')]
    adc_sup_pick_red['sup_champ_deaths'][i]=participants_output_if_red['deaths'][i][participantextendedstats_output_red['position'][i].index('SUPPORT')]
    adc_sup_pick_red['sup_champ_assists'][i]=participants_output_if_red['assists'][i][participantextendedstats_output_red['position'][i].index('SUPPORT')]

del adc_sup_pick_red['championid']


# In[778]:


adsup_pick=pd.concat([adc_sup_pick_blue,adc_sup_pick_red])
adsup_pick.reset_index(inplace=True)
del adsup_pick['index']


# In[779]:


adsup_pick.head()


# In[793]:


adsup_pickrate=pd.DataFrame(adsup_pick.groupby(['adc_champ_name','sup_champ_name']).count()['adc_champ']).sort_values(by='adc_champ',axis=0,ascending=False)


# In[795]:


adsup_pickrate['duo_pickrate']=''
adsup_pickrate['duo_pickrate']=adsup_pickrate['adc_champ']/len(sample)


# In[796]:


adsup_pickrate=adsup_pickrate.rename({'adc_champ':'duo_pickcount'},axis='columns')


# In[797]:


adsup_pickrate.reset_index(inplace=True)


# In[804]:


adsup_pickrate


# ### 승률 계산

# In[824]:


adsup_winrate=adsup_pick
adsup_winrate = adsup_winrate.astype({'win': 'str'})
adsup_winrate["win"] = adsup_winrate["win"].apply(lambda x: 1 if x=="['Win']" else 0)


# In[825]:


adsup_winrate=adsup_winrate.groupby(['adc_champ_name','sup_champ_name']).mean().sort_values(by='win',axis=0,ascending=False)


# In[826]:


adsup_winrate
adsup_winrate.reset_index(inplace=True)
adsup_winrate=adsup_winrate.rename({'win':'duo_winrate'},axis='columns')


# In[827]:


adsup_winrate


# ### KDA 평균 계산

# In[1170]:


adsup_kdamean=adsup_pick


# In[1171]:


adsup_kdamean['adc_kda']=''
adsup_kdamean['sup_kda']=''


# In[ ]:


for i in range(0,100):
    if adsup_kdamean['adc_champ_deaths'][i]!=0:
        adsup_kdamean['adc_kda'][i]=(adsup_kdamean['adc_champ_kill'][i]+adsup_kdamean['adc_champ_assists'][i])/adsup_kdamean['adc_champ_deaths'][i]        
    else:
        adsup_kdamean['adc_kda'][i]=(adsup_kdamean['adc_champ_kill'][i]+adsup_kdamean['adc_champ_assists'][i])*1.2
    
    if adsup_kdamean['sup_champ_deaths'][i]!=0:
        adsup_kdamean['sup_kda'][i]=(adsup_kdamean['sup_champ_kill'][i]+adsup_kdamean['sup_champ_assists'][i])/adsup_kdamean['sup_champ_deaths'][i]
    else:
        adsup_kdamean['sup_kda'][i]=(adsup_kdamean['sup_champ_kill'][i]+adsup_kdamean['sup_champ_assists'][i])*1.2
    


# In[1172]:


adsup_kdamean['duo_kda']=(adsup_kdamean['adc_kda']+adsup_kdamean['sup_kda'])/2


# In[ ]:


adsup_kdamean.head()


# In[859]:


adsup_kdamean = adsup_kdamean.astype({'duo_kda': 'int'})


# In[860]:


adsup_kdamean=adsup_kdamean.groupby(['adc_champ_name','sup_champ_name']).mean()


# In[861]:


adsup_kdamean.reset_index(inplace=True)


# In[862]:


adsup_kdamean


# ## 데이터 합치기

# In[867]:


adsup_stat=pd.merge(adsup_winrate,adsup_pickrate,on=['adc_champ_name','sup_champ_name'])
adsup_stat=pd.merge(adsup_stat,adsup_kdamean,on=['adc_champ_name','sup_champ_name'])
adsup_stat=adsup_stat.rename({'win':'duo_winrate'},axis='columns')


# ## 최종 데이터

# In[894]:


adsup_stat


# In[ ]:





# # 데이터 분석

# ## 주제 : 픽률, 승률, KDA를 기준으로 한 바텀듀오 티어
# 1. 가장 많이 픽된 듀오는?
# 2. 가장 승률이 높은 듀오는?
# 3. 가장 티어 (픽률, 승률, KDA 기준) 가 높은 듀오는?

# LOL을 하면서 뜻하지 않게 서포터나 원거리 딜러 포지션에 배정받을 경우가 있다. 바텀 듀오는 바텀 라인에서 마치 둘이 한 몸이 된 것처럼 행동해야 CS나 적 챔피언을 잡을 수 있고 이는 곧 게임의 승패까지 연관된다.
# 이 분석을 통해 고랭크 유저들의 바텀듀오 조합을 바탕으로 티어를 책정했으며, 바텀 유저가 아니거나 저랭크 유저들이 데이터 분석 결과를 바탕으로 조합을 구성하는데 조금 더 도움이 됐으면 하는 목적이다.

# 사용데이터
# - teams_output_blue , red : 팀에 대한 정보 (밴, 오브젝트 등)
# - participants_output_if_blue , red : 플레이어 챔피언에 대한 정보
# - gameduration : 게임 시간
# - participantextendedstats_output_blue , red : 플레이어 포지션, 티어
# - ch_if_df : 챔피언 정보
# 

# ## 1. 가장 많이 픽된 듀오는?

# In[874]:


adsup_stat['duo_pickcount'].mean()


# In[871]:


plt.hist(adsup_stat['duo_pickcount'])


# - LOL에는 수많은 원딜과 서포터들이 있고, 그 중 비원딜이나 다른 라인에서 잠시 내려온 서포터도 존재하기 때문에 모든 경우의 수를 고려할 수가 없다.
# - 너무 데이터가 없는 값은 삭제하기로 한다.
# - 평균 66회의 듀오픽이지만, 데이터가 좌측에 극단적으로 몰려있으므로, 임의적으로 300회 이상의 듀오 카운트를 가진 데이터만 사용하고자 한다.
# 

# In[910]:


adsup_stat_cut=adsup_stat[adsup_stat['duo_pickcount']>1000]


# In[911]:


plt.hist(adsup_stat_cut['duo_pickcount']) # 나름 고른 모습을 보여줌.


# In[913]:


len(adsup_stat_cut) # 17만건 게임의 95개의 듀오 대상


# In[914]:


adsup_stat_cut.sort_values(by='duo_pickcount',axis=0,ascending=False)[0:10]


# - 이즈리얼, 유미 듀오가 가장 많이 나왔다. 유미는 원거리 딜러의 몸에 달라붙어 있기 때문에 생존기가 우월한 이즈리얼이 자주 등장한다.
# - 그 뒤로 케이틀린, 럭스 & 케이틀린, 모르가나이다. 케이틀린은 덫을 활용하여 헤드샷을 최대한 쏴 라인전을 강하게 가져가야 하기 때문에 원거리 속박기가 있는 럭스, 모르가나가 선호되었다.
# - 그 뒤로 이즈리얼, 카르마이다. 역시 하드포킹 조합이다.
# - 모두 솔로랭크에서 자주 볼 수 있는 조합이며, 승률도 대부분 50%를 넘는 모습을 보여주었다.

# ## 2. 가장 승률이 높은 듀오는?
# 

# In[915]:


adsup_stat_cut.sort_values(by='duo_winrate',axis=0,ascending=False)[0:10]


# - 애쉬, 노틸러스는 CC에 맞을경우 한방에 갈 확률이 매우 높은 듀오이다. 두 챔프 모두 많은 CC기를 보유하고 있으며, 애쉬의 긴 사거리와 노틸러스의 닻줄을 통해 라인전을 강하게 가져간다.
# - 진, 세나는 원거리 딜링, CC지원으로 상체의 캐리를 돕는 픽이다. 역시 궁합이 좋은 편이라고 할 수 있따.
# - 루시안, 파이크는 사거리는 짧지만 파이크의 CC가 한번 닿을 경우, 상대 듀오를 모두 잡아낼 수 있는 픽이다.
# - 그 외에 솔로랭크에서 궁합이 좋은 챔피언들이 구성되었다.

# ## 3. 가장 티어 (픽률, 승률, KDA 기준) 가 높은 듀오는?
# - 듀오 픽률, 듀오 승률, 듀오 KDA를 기준으로 티어를 5티어까지 나눠보고자 한다.
# - 세 변수를 기준으로 K-means 군집분석을 실시하여 군집을 나눈 후, 군집에 따라 티어를 책정했다.

# In[1121]:


adsup_stat_clustering=adsup_stat_cut[['adc_champ_name','sup_champ_name','duo_winrate','duo_pickrate','duo_kda']]
adsup_stat_clustering.reset_index(inplace=True)
del adsup_stat_clustering['index']
adsup_stat_clustering


# In[1109]:


X=np.array(adsup_stat_clustering.iloc[:,2:5])
X[0:5]


# In[1110]:


scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X)


# In[1111]:


adsup_stat_clustering


# In[1144]:


X=pd.DataFrame(X_train_scale,columns=adsup_stat_clustering.columns[2:5])
X.head()


# In[1113]:


model = KMeans(n_clusters=5,algorithm='auto')
feature = X[['duo_winrate','duo_pickrate','duo_kda']]


# In[1114]:


model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns=['predict']


# In[1115]:


adsup_stat_clustering_output=pd.concat([adsup_stat_clustering,predict], axis=1)


# In[1143]:


adsup_stat_clustering_output.head()


# In[1118]:


count=adsup_stat_clustering_output.groupby('predict').count()['adc_champ_name']
count 


# In[1119]:


X=pd.concat([X,predict],axis=1)


# In[1128]:


X.groupby('predict').mean().mean(axis=1)
# 2 -> 1티어, 4 -> 2티어, 0 -> 3티어, 3 -> 4티어, 4 -> 5티어


# In[1147]:


adsup_stat_clustering_output['tier']=''
for i in range(len(adsup_stat_clustering_output)):
    if adsup_stat_clustering_output['predict'][i]==2:
        adsup_stat_clustering_output['tier'][i]='1티어'
    elif adsup_stat_clustering_output['predict'][i]==4:
        adsup_stat_clustering_output['tier'][i]='2티어'
    elif adsup_stat_clustering_output['predict'][i]==0:
        adsup_stat_clustering_output['tier'][i]='3티어'
    elif adsup_stat_clustering_output['predict'][i]==3:
        adsup_stat_clustering_output['tier'][i]='4티어'
    else:
        adsup_stat_clustering_output['tier'][i]='5티어'
del adsup_stat_clustering_output['predict']


# In[1156]:


adsup_stat_clustering_output.head()


# In[1160]:


adsup_stat_clustering_output[adsup_stat_clustering_output['tier']=='1티어'] #1티어 바텀듀오


# - 1티어 바텀듀오는 위와 같다.
# - 대부분의 경기에서 등장했으며, 승률과 픽률 모두 높은 양상을 띈다.
# - 솔로랭크에서 대부분 한 번쯤 봤던 조합이며, 이따금 대회에서 나오기도 하는 조합이다.

# In[1161]:


adsup_stat_clustering_output[adsup_stat_clustering_output['tier']=='2티어'] #2티어 바텀듀오


# In[1162]:


adsup_stat_clustering_output[adsup_stat_clustering_output['tier']=='3티어'] #3티어 바텀듀오


# In[1163]:


adsup_stat_clustering_output[adsup_stat_clustering_output['tier']=='4티어'] #4티어 바텀듀오


# In[1164]:


adsup_stat_clustering_output[adsup_stat_clustering_output['tier']=='5티어'] #5티어 바텀듀오


# - 5티어 바텀조합이다. 주의해야 할 것은 1000회 이상 데이터가 기록된 데이터를 바탕으로 분석을 진행한 것이기 때문에 이 조합이 꼭 나쁘다는 것은 아니다.
# - 상위티어 챔피언 구성에 비해 다소 지표가 떨어진다.

# # 한계점
# - 듀오승률, 듀오픽률, KDA 만 고려했을 뿐, 데미지 딜링이나 골드 수급 등 많은 변수를 고려하지 않은 분석이다. 특히 KDA라는 지표는 허점이 많은 지표이기 때문에 보정이나 다른 데이터 대체가 필요할 수도 있다.
# - 픽률이 군집분석에서 너무 높은 부분을 가져 간듯 하다. 케이틀린, 이즈리얼같은 국민픽이라고 해서 반드시 상위티어 일수는 없다.
# - 다른 고차원적인 분석이 분명 존재할 것이다.

# # 보완해야 할 점
# - JSON 파일을 다루는 데에 좀 더 익숙해져야 한다. JSON 파일 로드에 굉장히 많은 시간을 쏟았고, 결국 내부를 임의적으로 바꿔 로드하는데 그치고 말았다. 결과위주의 분석이기 때문에 추후 코드최적화가 반드시 필요하다.
# - 리그오브레전드에 대한 이해도가 더욱 필요하다. 게임 상세정보에 분명 더 좋은 인사이트를 추출할 부분이 있을 것이다.
