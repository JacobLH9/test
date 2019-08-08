# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:54:35 2019

@author: LEEJ001
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
######Getting forward data
file=r'C:\Users\LEEJ001\Desktop\Master_Scripts\RiskProject\CleanData\FX Fwrd points - USDCADFormatted.xlsx'
df=pd.read_excel(file,sheet_name=[1])[1]
cols=[]
columns=list(df.columns)
for i in range(0,104):
    if str(columns[i])[0]=='C':
        cols.append(columns[i])
for i in cols:
    df[i]=pd.to_numeric(df[i])
    df[i]=(pd.TimedeltaIndex(df[i],unit='d')+datetime(1900,1,1)).strftime('%Y-%m-%d')
rows=pd.Index.tolist(df)

spotA={}
spotB={}
for i in range(1,len(rows)):
    spotA[rows[i][0]]=rows[i][3]
    spotB[rows[i][0]]=rows[i][2]
forA={}
forB={}

days=list(df['CAD4Y Curncy'])[1:]
for j in range(4,104):
    for i in range(1,len(rows)):
        if rows[i][j] not in forA.keys() and columns[j] in cols and rows[i][j] in days:
            forA[rows[i][j]]=[]
            forA[rows[i][j]].append(rows[i][j+3])
        if rows[i][j] not in forB.keys() and columns[j] in cols and rows[i][j] in days:
            forB[rows[i][j]]=[]
            forB[rows[i][j]].append(rows[i][j+2])   
        elif columns[j] in cols and rows[i][j] in days:
            forA[rows[i][j]].append(rows[i][j+3])
            forB[rows[i][j]].append(rows[i][j+2])
p=[]
for i in forA.keys():
    if len(forA[i])<16:
        p.append(i)
for i in range(0,len(p)):
    try:
        days.remove(p[i])
    except:
        continue
####GETTING OIS DATA
path = pd.ExcelFile(r'C:\Users\LEEJ001\Desktop\Master_Scripts\RiskProject\CleanData\USDCAD Basis input.xlsx')
caois = pd.read_excel(path, 'CAD OIS')
usois = pd.read_excel(path, 'USD OIS')
xs=list(caois['M_DAYSNB'])[:17]
xs.remove(270)
cois={}
uois={}


for i in range(0,len(caois)):
    d=str(caois.iat[i,0])
    key= d[:4]+'-'+d[4:6]+'-'+d[6:]
    if key not in cois.keys():
        cois[key]=[]
        cois[key].append(caois.iat[i,2])
    elif caois.iat[i,1]<=3600 and caois.iat[i,1]!=270:
        cois[key].append(caois.iat[i,2])
for i in range(0,len(usois)):
    d=str(usois.iat[i,0])
    key= d[:4]+'-'+d[4:6]+'-'+d[6:]
    if key not in uois.keys():
        uois[key]=[]
        uois[key].append(usois.iat[i,2])
    elif usois.iat[i,1]<=3600 and usois.iat[i,1]!=270:
        uois[key].append(usois.iat[i,2])
uois.pop('2012-05-31', None)

#####GETTING RIGHT CADOIS DATA
fois = pd.read_excel(r'C:\Users\LEEJ001\Desktop\Master_Scripts\RiskProject\CleanData\cad ois.xlsx')
ois={}
for i in range(0,10650):
    if fois.iat[i,2].strftime('%Y-%m-%d') not in ois.keys():
        ois[fois.iat[i,2].strftime('%Y-%m-%d')]=[]
        ois[fois.iat[i,2].strftime('%Y-%m-%d')].append(fois.iat[i,3])
    elif fois.iat[i,4]!=120 and fois.iat[i,4]!=150 and fois.iat[i,4]!=456 and fois.iat[i,4]!=270:
        ois[fois.iat[i,2].strftime('%Y-%m-%d')].append(fois.iat[i,3])

for i in list(ois.keys())[184:]:
    p=cois[i]
    q=ois[i]
    z=[]
    for j in range(0,8):
        z.append(p[j])
    for j in range(7,len(q)-1):
        z.append(q[j])
    cois[i]=z


##### CREATING BASIS CURVES
def basis(spot,forw,ca,us,term):
    x=spot+forw/10000
    y=np.log(x/spot)*10000/(term/360)
    z=(ca-us)*100
    return z-y
basA={}
basB={}
def curver(spots,forws,cas,uss,terms,days):
    dic={}
    for i in days:
        try:
            spot=spots[i]
            curve=[]
            for j,k,l,m in zip(forws[i],cas[i],uss[i],terms):
                curve.append(basis(spot,j,k,l,m))
                dic[i]=curve
        except:
            continue
    return dic
basA=curver(spotA,forA,cois,uois,xs,days)
basB=curver(spotB,forB,cois,uois,xs,days)

####FIXING LONG END OF BASIS CURVE
basis=pd.read_excel(r'C:\Users\LEEJ001\Desktop\Master_Scripts\RiskProject\CleanData\usdcadbasis.xlsx')
cols=list(basis.columns)
cols=sorted(np.array(list(basis.columns)).astype(float))
cols=[2,'2.1',3,'3.1',4,'4.1',5,'5.1',6,'6.1',7,'7.1',8,'8.1',9,'9.1',10,'10.1',12,'12.1',15,'15.1',20,'20.1',30,'30.1']
basis=basis[cols]
ind=basis.index.values.tolist()
dicA={}
dicB={}
for i in range(2,len(basis)):
    row=list(basis.iloc[i])
    dicA[ind[i].strftime('%Y-%m-%d')]=row[1::2]
    dicB[ind[i].strftime('%Y-%m-%d')]=row[::2]
finalA={}
finalB={}
for i in dicA.keys():
    try:
        pa=basA[i][:7]
        pb=basB[i][:7]
        qa=dicA[i][:9]
        qb=dicB[i][:9]
        finalA[i]=pa+qa
        finalB[i]=pb+qb
    except:
        continue


basM={}
basS={}
for i in basA.keys():
   mids=[]
   spreads=[]
   for j,k in zip(basB[i],basA[i]):
       mids.append(np.mean([j,k]))
       spreads.append(abs(j-k))
   basM[i]=mids
   basS[i]=spreads
weights={}
for i in basS.keys():
    arr=np.array(basS[i])
    weights[i]=1/(arr)**2
k=360
xs=list(caois['M_DAYSNB'])[:12]
xs.remove(270)

fxs=[]
for t in xs:
    fxs.append((t-k)/(t+k))
    
def Cheb2poly(f,x):
    ys=[]
    for i in x:
        y=0
        for j in range(0,len(f)):
            y+=((i)**(j))*(f[j])
        ys.append(y)
    return ys


exclude=['2012-10-04','2015-06-11']
for i in basM.keys():
   if len(basM[i])<16:
       exclude.append(i)
keys=list(basM.keys())
keys=[x for x in keys if x not in exclude]
x=np.array(fxs)
basF={}
for i in range(0,len(keys)):
    if i%100==0:
        print(keys[i])
    y=basM[keys[i]]
    yerr=basS[keys[i]]
    try:
        plt.errorbar(x[1:], y[1:], yerr=yerr[1:], label='Data')
        fit=np.polynomial.Chebyshev.fit(fxs[1:],basM[keys[i]][1:],deg=3,w=weights[keys[i]][1:])
        plt.plot(x[1:],fit(x)[1:],label='Fit')
        basF[keys[i]]=fit(x)
        plt.legend()
        plt.title(keys[i])
        path=(r"C:\Users\LEEJ001\Desktop\Fits\'{}'.png").format(keys[i])
#        plt.show()
        plt.savefig(path)
        plt.clf()
    except:
        plt.clf()
        continue
fR={}
dR={}
keys=list(basF.keys())
for i in range(0,len(basF.keys())-1):
    retsD=[]
    pF=np.array(basF[keys[i]])
    aF=np.array(basF[keys[i+1]])
    fR[keys[i+1]]=aF-pF
    pD=np.array(basM[keys[i]])
    aD=np.array(basM[keys[i+1]])
    dR[keys[i+1]]=aD-pD



xs=[]
ys=[]
for i in range(0,15):
    x=[]
    y=[]
    for j in list(fR.keys()):
        x.append(dR[j][i])
        y.append(fR[j][i])
    xs.append(x)
    ys.append(y)


titles=['ON','1M','2M','3M','6M','1Y','1.5Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y']
for i,j,k in zip(xs,ys,titles):
    plt.scatter(i,j)
    plt.title(k)
    plt.xlabel('Changes in mids')
    plt.ylabel('Changes in fit')
    path=(r"C:\Users\LEEJ001\Desktop\ReturnPlots\'{}'.png").format(k)
    plt.savefig(path)
    plt.clf()
    
volsF=[]
volsM=[]
for i,j in zip(xs,ys):
    volsM.append(np.nanstd(i))
    volsF.append(np.nanstd(j))

wilds=[]
for i in fR.keys():
    for j,k in zip(fR[i],dR[i]):
        if abs(j)>30+abs(k) and i not in wilds and abs(j)>20 and int(i[:4])>2010:
            wilds.append(i)

