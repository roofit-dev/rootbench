import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
times = pd.read_csv('testUnBinned03.csv',skiprows=3)
times = times[['name','real_time','cpu_time']]

times['name'] = times['name'].map(lambda x: x.replace('/iterations:12/real_time',''))
times['name'] = times['name'].map(lambda x: x.replace('BM_RooFit_',''))
times['process'] = times['name'].map(lambda x: x.split("/")[0])
times['events'] = times['name'].map(lambda x: int(x.split("/")[1]))
times['ncpus'] = times['name'].map(lambda x: int(x.split("/")[2]))
times['ideal'] = times['real_time']/times['ncpus']

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
h = sns.factorplot(x="events", y="time",hue="ncpus",col="process", data = types, type='bar')
h.set_axis_labels("Number of Events", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)
times = plottimes[plottimes['type']!='cpu']
times = times[times['process']=='BDecayWithMixing']
g = sns.factorplot(x="events", y="time",hue="ncpus",col="type", data = times, type='bar')
g.set_axis_labels("Number of Events", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)

cputimes = plottimes[plottimes['ncpus']>1]
f = sns.factorplot(x="events", y="time",hue="ncpus",col="type", data = cputimes[cputimes['type']=='cpu'], type='bar')
f.set_axis_labels("Number of Events", "Time (Seconds)").despine(left=True).set_xticklabels(rotation=45)


plt.show()
