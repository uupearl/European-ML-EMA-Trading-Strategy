
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE=Path(__file__).resolve().parent
FIG=BASE.parent/'Results'/'Figures'; FIG.mkdir(parents=True,exist_ok=True)
RES=BASE.parent/'Results'
RAW=BASE.parent/'Data'/'European_ETF_Daily_2021_2025.csv'
d=pd.read_csv(RAW,parse_dates=['Date']).sort_values(['Ticker','Date'])
prices=d.pivot(index='Date',columns='Ticker',values='Adj Close').sort_index()
r=prices.pct_change(fill_method=None)
perf=pd.read_csv(RES/'performance_summary_2025.csv',index_col=0)
ret=pd.read_csv(RES/'portfolio_daily_returns_2025.csv',index_col=0,parse_dates=True)
imp=pd.read_csv(RES/'feature_importance.csv')
thr=pd.read_csv(RES/'threshold_tuning_2024.csv')
auc=pd.read_csv(RES/'auc_by_ticker.csv')
pred=pd.read_csv(RES/'test_2025_predictions_and_signals.csv',parse_dates=['Date'])
TICKERS=sorted(d['Ticker'].unique())
COLORS=['#1565C0','#E67E22','#00897B','#7E57C2','#C62828','#D4A017','#5C6BC0','#43A047','#C05E91','#546E7A']
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Tinos','DejaVu Serif'],'axes.edgecolor':'#888888','axes.labelcolor':'black','text.color':'black','xtick.color':'black','ytick.color':'black','savefig.facecolor':'white','figure.facecolor':'white','font.size':10,'grid.color':'#dedede','lines.linewidth':1.7})

def fmt_ax(ax,grid=True):
 ax.spines[['top','right']].set_visible(False)
 if grid: ax.grid(alpha=.65, linewidth=.6)
 ax.set_axisbelow(True)

def save(name):
 plt.tight_layout(pad=1.4);plt.savefig(FIG/name,dpi=190,bbox_inches='tight',facecolor='white');plt.close()

# 1 normalized market index
fig,ax=plt.subplots(figsize=(11,5.4));norm=prices.div(prices.iloc[0]).mul(100)
for i,c in enumerate(TICKERS): ax.plot(norm.index,norm[c],color=COLORS[i],linewidth=(2.3 if c in ('EWG','EWQ','EWU') else 1.15),label=c,ls=['-','--','-.',':'][i%4])
fmt_ax(ax); ax.set_title('European ETF Market Landscape | Adjusted Prices (2021 = 100)',weight='bold');ax.set_ylabel('Normalized price index');ax.legend(ncol=5,frameon=False,fontsize=9,loc='upper left');save('01_market_normalized_prices.png')

# 2 2025 return bar
p25=(prices[prices.index.year==2025].iloc[-1]/prices[prices.index.year==2025].iloc[0]-1).sort_values()
fig,ax=plt.subplots(figsize=(9,4.8));bars=ax.barh(p25.index,100*p25.values,color='#1976D2');ax.axvline(0,color='black',lw=1);fmt_ax(ax);ax.set_xlabel('2025 adjusted-price return (%)');ax.set_title('Country ETF Performance in 2025',weight='bold')
for bar,v in zip(bars,p25): ax.text(bar.get_width()+1 if v>=0 else bar.get_width()-5,bar.get_y()+bar.get_height()/2,f'{100*v:+.1f}%',va='center',fontsize=8)
save('02_etf_2025_returns.png')

# 3 annual returns matrix
ann=prices.resample('YE').last().pct_change(fill_method=None)
# For first year use first raw price as starting baseline, still 2021 first date not Dec 2020
ann.loc[pd.Timestamp('2021-12-31')]=prices.loc['2021'].iloc[-1]/prices.loc['2021'].iloc[0]-1
ann=ann.sort_index().loc['2021':'2025',TICKERS]
vals=ann.T*100
fig,ax=plt.subplots(figsize=(9,5.4));im=ax.imshow(vals.values,aspect='auto',cmap='RdBu_r',vmin=-55,vmax=55);ax.set_yticks(range(len(TICKERS)),TICKERS);ax.set_xticks(range(5),[str(x.year) for x in ann.index]);ax.set_title('Annual ETF Price Returns | 2021–2025 (%)',weight='bold');
for i in range(vals.shape[0]):
 for j in range(vals.shape[1]):
  v=vals.iloc[i,j]; ax.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=8,color=('white' if abs(v)>31 else 'black'))
fig.colorbar(im,ax=ax,shrink=.72,label='Annual return (%)');save('03_annual_return_matrix.png')

# 4 correlation 2021–2025
corr=r.corr().loc[TICKERS,TICKERS]
fig,ax=plt.subplots(figsize=(8,6.2));im=ax.imshow(corr.values,cmap='YlGnBu',vmin=.60,vmax=1);ax.set_xticks(range(10),TICKERS,rotation=45);ax.set_yticks(range(10),TICKERS);ax.set_title('Correlation Matrix | Daily ETF Returns',weight='bold');
for i in range(10):
 for j in range(10):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7.6,color='white' if corr.iloc[i,j]>.82 else 'black')
fig.colorbar(im,ax=ax,shrink=.74);save('04_return_correlation_heatmap.png'); corr.to_csv(RES/'return_correlation_2021_2025.csv')

# 5 price EMA example
s=d[d.Ticker=='EWG'].set_index('Date').sort_index()['Adj Close'];e10=s.ewm(span=10,adjust=False).mean();e50=s.ewm(span=50,adjust=False).mean()
fig,ax=plt.subplots(figsize=(10,5));sample=s.loc['2025'];ax.plot(sample.index,sample.values,color='#233142',label='EWG Adjusted Close',lw=1.7);ax.plot(sample.index,e10.loc['2025'],color='#E67E22',label='EMA 10',lw=1.3,linestyle='--');ax.plot(sample.index,e50.loc['2025'],color='#00897B',label='EMA 50',lw=1.8,linestyle='-.');fmt_ax(ax);ax.legend(frameon=False,ncol=3);ax.set_title('Traditional EMA Direction | EWG in 2025',weight='bold');ax.set_ylabel('USD-adjusted price');save('05_ema_example_EWG.png')

# 6 normalized code results feature importance
fig,ax=plt.subplots(figsize=(8,5.7));tmp=imp.iloc[:10].iloc[::-1];ax.barh(tmp.Feature,100*tmp.Importance,color='#1565C0');fmt_ax(ax);ax.set_xlabel('Mean decrease in impurity importance (%)');ax.set_title('RF Feature Importance | Training Only (2021–2023)',weight='bold');save('06_rf_feature_importance.png')

# 7 RF AUC by ticker
fig,ax=plt.subplots(figsize=(8,4.7));auc2=auc.sort_values('Test_AUC_2025');ax.barh(auc2.Ticker,auc2.Test_AUC_2025,color='#7E57C2');ax.axvline(.5,color='black',linestyle='--',lw=1.4,label='Random classifier 0.50');ax.set_xlim(.0,.65);fmt_ax(ax);ax.set_xlabel('2025 ROC-AUC');ax.set_title('Out-of-Sample Prediction: Weak and Uneven AUC',weight='bold');ax.legend(frameon=False,loc='lower right');save('07_test_auc_by_etf.png')

# 8 threshold validation
fig,ax=plt.subplots(figsize=(8.8,4.4));ax.plot(thr.Threshold,thr['Validation Sharpe'],color='#1565C0',marker='o',ms=3,lw=1.3);ax.axvline(.58,color='#E67E22',ls='--',label='Original paper: 0.58');ax.axvline(.59,color='#00897B',ls=':',label='Selected: 0.59');ax.axhline(0,color='#777777',lw=.65);fmt_ax(ax);ax.set_title('Validation-Only Threshold Search (2024)',weight='bold');ax.set_ylabel('Validation Sharpe');ax.set_xlabel('Confidence cutoff');ax.legend(frameon=False);save('08_threshold_validation.png')

# 9 portfolio cumulative returns
fig,ax=plt.subplots(figsize=(10.6,5.5))
for i,c in enumerate(ret.columns):
 ax.plot(ret.index,(1+ret[c]).cumprod(),color=COLORS[i],ls=['-','--','-.',':','-'][i],lw=1.9 if i in (0,3) else 1.5,label=c.replace('_',' '))
ax.axhline(1,color='#E67E22',lw=.7);fmt_ax(ax);ax.set_title('2025 Out-of-Sample Equity Curves | $1 Initial Capital',weight='bold');ax.set_ylabel('Portfolio wealth');ax.legend(frameon=False,ncol=2,fontsize=8.8);save('09_portfolio_cumulative_returns.png')

# 10 drawdown
fig,ax=plt.subplots(figsize=(10.6,5.2))
for i,c in enumerate(ret.columns):
 wealth=(1+ret[c]).cumprod(); dd=wealth/wealth.cummax()-1
 ax.plot(ret.index,dd*100,color=COLORS[i],ls=['-','--','-.',':','-'][i],lw=1.8,label=c.replace('_',' '))
fmt_ax(ax);ax.set_title('2025 Maximum-Drawdown Paths',weight='bold');ax.set_ylabel('Drawdown (%)');ax.legend(frameon=False,ncol=2,fontsize=8.8);save('10_portfolio_drawdowns.png')

# 11 table-like performance chart with mdd and annual return
fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(len(perf));w=.27
ax.bar(x-w,perf['Total Return']*100,width=w,label='Total return',color='#1565C0')
ax.bar(x,perf['Annualized Volatility']*100,width=w,label='Annualized vol.',color='#E67E22')
ax.bar(x+w,-perf['Maximum Drawdown']*100,width=w,label='Max drawdown (absolute)',color='#00897B',edgecolor='#00897B',linewidth=.4)
ax.set_xticks(x,['Equal weight','Traditional EMA','Fixed 0.58','Tuned 0.59','Dynamic Q90'],rotation=12,ha='right');fmt_ax(ax);ax.set_ylabel('Percent (%)');ax.set_title('Return, Risk, and Drawdown Across Strategies',weight='bold');ax.legend(frameon=False,ncol=3);save('11_risk_return_bars.png')

# 12 strategy signal illustration EWG 2025
s=pred[pred.Ticker=='EWG'].sort_values('Date')
fig,ax=plt.subplots(figsize=(10,5)); ax.plot(s.Date,s['Adj Close'],color='#233142',lw=1.4,label='EWG price');long=s.ml_tuned_signal>1e-10;short=s.ml_tuned_signal<-1e-10
ax.scatter(s.loc[long,'Date'],s.loc[long,'Adj Close'],marker='^',color='#2E7D32',s=24,label='ML approved long');ax.scatter(s.loc[short,'Date'],s.loc[short,'Adj Close'],marker='v',facecolors='#EF5350',edgecolors='#C62828',s=24,label='ML approved short');fmt_ax(ax);ax.set_title('EWG in 2025 | EMA Signals Retained by RF Filter',weight='bold');ax.legend(frameon=False,ncol=3,fontsize=8.8);save('12_filtered_signals_EWG.png')

# 13 cost sensitivity using original daily weights + forward returns
p=pred.pivot(index='Date',columns='Ticker',values='fwd_ret_1d').fillna(0).sort_index()
strategy_paths={
 'Traditional EMA':'weights_traditional_ema.csv',
 'ML Fixed 0.58':'weights_ml_fixed_058.csv',
 'ML Tuned 0.59':'weights_ml_tuned.csv',
 'ML Dynamic Q90':'weights_ml_dynamic.csv',
}
rows=[];rob=[]
for name,file in strategy_paths.items():
 wgt=pd.read_csv(RES/file,index_col=0,parse_dates=True).reindex(index=p.index,columns=p.columns).fillna(0)
 turnover=.5*wgt.diff().abs().sum(axis=1)
 turnover.iloc[0]=.5*wgt.iloc[0].abs().sum()
 gross=(wgt*p).sum(axis=1)
 for bps in [0,5,10,20]:
  rn=gross-turnover*bps/10000
  rows.append({'Strategy':name,'Cost_bps':bps,'Total_Return':(1+rn).prod()-1,'Sharpe':rn.mean()/rn.std(ddof=1)*np.sqrt(252),'Annualized_Volatility':rn.std(ddof=1)*np.sqrt(252)})
 # strict dollar neutral from the raw signals (previously generated, unchanged selection)
 col={'Traditional EMA':'ema_signal','ML Fixed 0.58':'ml_fixed_058_signal','ML Tuned 0.59':'ml_tuned_signal','ML Dynamic Q90':'ml_dynamic_signal'}[name]
 signal=pred.pivot(index='Date',columns='Ticker',values=col).reindex(index=p.index,columns=p.columns).fillna(0)
 longs=signal.clip(lower=0); shorts=-signal.clip(upper=0)
 longsum=longs.sum(axis=1).replace(0,np.nan);shortsum=shorts.sum(axis=1).replace(0,np.nan)
 ws=(.5*longs.div(longsum,axis=0)-.5*shorts.div(shortsum,axis=0)).fillna(0)
 mask=longsum.notna()&shortsum.notna();ws.loc[~mask,:]=0
 turn=.5*ws.diff().abs().sum(axis=1);turn.iloc[0]=.5*ws.iloc[0].abs().sum()
 rr=(ws*p).sum(axis=1)-turn*5/10000
 rob.append({'Strategy':name,'Portfolio_Mode':'strict_dollar_neutral','Total_Return':(1+rr).prod()-1,'Sharpe':rr.mean()/rr.std(ddof=1)*np.sqrt(252) if rr.std()>0 else np.nan,'Max_Drawdown':((1+rr).cumprod()/(1+rr).cumprod().cummax()-1).min(),'Nonzero_Trading_Days':int(mask.sum())})
for name,row in perf.iterrows():
 if name=='EqualWeight_DailyRebalanced':continue
 rob.append({'Strategy':{'Traditional_EMA':'Traditional EMA','ML_EMA_Fixed_058':'ML Fixed 0.58','ML_EMA_Tuned_0.59':'ML Tuned 0.59','ML_EMA_Dynamic_Q90':'ML Dynamic Q90'}[name], 'Portfolio_Mode':'gross_normalized','Total_Return':row['Total Return'],'Sharpe':row['Sharpe Ratio'],'Max_Drawdown':row['Maximum Drawdown'],'Nonzero_Trading_Days':np.nan})
cs=pd.DataFrame(rows);cs.to_csv(RES/'transaction_cost_sensitivity_2025.csv',index=False)
rb=pd.DataFrame(rob);rb.to_csv(RES/'strict_dollar_neutral_robustness_2025.csv',index=False)
fig,ax=plt.subplots(figsize=(9,4.8))
for i,(name,group) in enumerate(cs.groupby('Strategy',sort=False)):
 ax.plot(group.Cost_bps,group.Total_Return*100,marker=['o','s','^','d'][i],ls=['-','--','-.',':'][i],color=COLORS[i],label=name)
fmt_ax(ax);ax.set_xlabel('One-way transaction cost (bps)');ax.set_ylabel('2025 cumulative return (%)');ax.set_title('Trading-Cost Sensitivity | Same Predictions and Weights',weight='bold');ax.legend(frameon=False,ncol=2,fontsize=8.6);save('13_cost_sensitivity.png')
fig,ax=plt.subplots(figsize=(9,4.7));subset=rb[rb.Portfolio_Mode=='strict_dollar_neutral'];labs=subset.Strategy.tolist();ax.barh(labs,subset.Total_Return*100,color='#7E57C2');fmt_ax(ax);ax.set_xlabel('2025 total return (%)');ax.set_title('Strict Dollar-Neutral Robustness | Net Returns',weight='bold');save('14_strict_neutral_robustness.png')

# 14 results summary CSV for assembly
print('Created figures:',len(list(FIG.glob('*.png'))))
print('Return correlation range (off diag):',float(corr.where(~np.eye(10,dtype=bool)).min().min()),float(corr.where(~np.eye(10,dtype=bool)).max().max()))
print('cost sensitivity:',cs[cs.Cost_bps==5][['Strategy','Total_Return']].to_string(index=False))
print('strict neutrality:',rb[rb.Portfolio_Mode=='strict_dollar_neutral'][['Strategy','Total_Return','Sharpe']].to_string(index=False))
