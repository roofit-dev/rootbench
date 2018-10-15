import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#./root/roofit/roofit/RoofitASUM --benchmark_out_format=csv --benchmark_out=NewerASUM.csv

times = pd.read_csv('NewerASUM.csv',skiprows=3)
times = times[['name','real_time','cpu_time']]

times['name'] = times['name'].map(lambda x: x.replace('/iterations:12/real_time',''))
times['name'] = times['name'].map(lambda x: x.replace('BM_RooFit_',''))
times['process'] = times['name'].map(lambda x: x.split("/")[0])
times['events'] = times['name'].map(lambda x: int(x.split("/")[1]))
times['ncpus'] = times['name'].map(lambda x: int(x.split("/")[2]))
times['ideal'] = times['real_time']/times['ncpus']
times['hf'] = times['process'].map(lambda x: True if 'Hist' in x else False)
times['quick'] = times['process'].map(lambda x: True if 'Stat' in x else False)
times['tiny'] = times['process'].map(lambda x: True if 'Short' in x else False)
times['BB'] = times['process'].map(lambda x: True if 'BB' in x else False)
print times
idealtimes = times.copy()
cputimes = times.copy()
idealtimes['type']='ideal'
cputimes['type']='cpu'
times['type']='real'
idealtimes['real_time']=idealtimes['ideal']
cputimes['real_time']=cputimes['cpu_time']

plottimes = times.append(idealtimes.append(cputimes))

plottimes['real_time']=plottimes['real_time']*10**-9
plottimes.rename(columns={'real_time':'time'},inplace = True)

print plottimes
types = plottimes[plottimes['type']=='real']
ASUM = types[types['hf']==False]

#h = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = ASUM, type='bar')
#h.set_axis_labels("Number of Bins", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)

HF = types[types['hf']==True]
long = HF[HF['quick']==False]
long = long[long['process']=='HistInterp']

#g = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = long, type='bar')
#g.set_axis_labels("Number of Bins", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)

short = HF[HF['quick']==True]
short = short[short['tiny']==False]
#i = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = short, type='bar')
#i.set_axis_labels("Number of Bins", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)

tiny = types[types['tiny']==True]
#j = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = tiny, type='bar')
#j.set_axis_labels("Number of Bins", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)

bb = types[types['BB']==True]
k = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = bb, type='bar')
k.set_axis_labels("Number of Bins", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)


plt.show()
