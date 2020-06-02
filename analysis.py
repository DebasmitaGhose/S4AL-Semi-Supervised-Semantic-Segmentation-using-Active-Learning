import numpy as np
import collections

s4gan_names = np.load('names_s4gan.npy')
uncert_names = np.load('uncertainity_sampling_names.npy')

print(np.shape(s4gan_names), 's4gan')
print(np.shape(uncert_names), 'uncert')

s4gan_names_02 = s4gan_names[:34]
s4gan_names_05 = s4gan_names[:85]
s4gan_names_125 = s4gan_names[:211]
s4gan_names_20 = s4gan_names[:338]
s4gan_names_33 = s4gan_names[:557]
s4gan_names_50 = s4gan_names[:845]
s4gan_names_75 = s4gan_names[:1267]


present_02 = np.in1d(uncert_names, s4gan_names_02)
ranks_02 = np.where(present_02==True)
#print(ranks_02, "2%")
print(str(np.count_nonzero(ranks_02[0]<34)) + " elements are certain out of 34 - 2%")

present_05 = np.in1d(uncert_names, s4gan_names_05)
ranks_05 = np.where(present_05==True)
#print(ranks_05, "5%")
print(str(np.count_nonzero(ranks_05[0]<85)) + " elements are certain out of 85 - 5%")

present_125 = np.in1d(uncert_names, s4gan_names_125)
ranks_125 = np.where(present_125==True)
#print(ranks_125, "12.5%")
print(str(np.count_nonzero(ranks_125[0]<211)) + " elements are certain out of 211 - 12.5%")

present_20 = np.in1d(uncert_names, s4gan_names_20)
ranks_20 = np.where(present_20==True)
#print(ranks_20, "20%")
print(str(np.count_nonzero(ranks_20[0]<338)) + " elements are certain out of 338 - 20%")

present_33 = np.in1d(uncert_names, s4gan_names_33)
ranks_33 = np.where(present_33==True)
#print(ranks_33, "33%")
print(str(np.count_nonzero(ranks_33[0]<557)) + " elements are certain out of 557 - 33%")

present_50 = np.in1d(uncert_names, s4gan_names_50)
ranks_50 = np.where(present_50==True)
#print(ranks_50, "50%")
print(str(np.count_nonzero(ranks_50[0]<845)) + " elements are certain out of 845 - 50%")


present_75 = np.in1d(uncert_names, s4gan_names_75)
ranks_75 = np.where(present_75==True)
#print(ranks_75, "75%")
print(str(np.count_nonzero(ranks_75[0]<1267)) + " elements are certain out of 1267 - 75%")





#print(ranks)
#print(collections.Counter(uncert_names[ranks]) == collections.Counter(s4gan_names_02))
