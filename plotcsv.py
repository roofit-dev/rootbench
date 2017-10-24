import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
times = pd.read_csv('vincescsv09.csv',skiprows=3)
times = times[['name','real_time','cpu_time']]

times['name'] = times['name'].map(lambda x: x.replace('/iterations:12/real_time',''))
times['name'] = times['name'].map(lambda x: x.replace('BM_RooFit_BinnedTest',''))
times['minimizer'] = times['name'].map(lambda x: x[:6])
migrad = lambda x: x if 'Migrad' in x else x[:5]
times['minimizer'] = times['minimizer'].map(migrad)
migradred = lambda x: x[6:] if 'Migrad' in x else x[5:]
times['name'] = times['name'].map(migradred)
nchanned = lambda x: int(x.split('/')[1]) if '_NChannel' in x else -1
times['channels'] = times['name'].map(nchanned)
rmnchanned = lambda x: x[len(x.split('/')[0]):].lstrip('/').rstrip('/').split('/')[1] if '_NChannel' in x else x
times['name'] = times['name'].map(rmnchanned) 
nbinned = lambda x: int(x[6:8].rstrip('/')) if '_NBin' in x else -1
times['bins'] = times['name'].map(nbinned)
rmbinned = lambda x: x[len(x.split('/')[0]):].lstrip('/').rstrip('/').split('/')[1] if '_NBin' in x else x
times['name'] = times['name'].map(rmbinned) 
times['name'] = times['name'].map(lambda x: int(x.lstrip('/')))
times.rename(columns={'name':'ncpus'}, inplace = True)
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

g = sns.factorplot(x="channels", y="time",hue="ncpus",col="type", data=plottimes[plottimes['channels']>0], type='bar')
g.set_axis_labels("number of channels", "Time (Seconds)").despine(left=True)

f = sns.factorplot(x="bins", y="time",hue="ncpus",col="type", data=plottimes[plottimes['bins']>0], type='bar')
f.set_axis_labels("number of bins", "Time (Seconds)").despine(left=True)

plt.show()
