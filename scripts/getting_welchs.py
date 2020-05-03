import os
import sys
sys.path.append("..")


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



def ttestPrint(model_1, model_2):
  print("Welch’s t_test p_values")
  print(
    stats.ttest_ind(model_1, model_2, equal_var = False)[1]
    )

#TABLE II NEW TRANSFORMS
hits_4_channels_PlusKernel_Transformer  = [ (0.99073675),  (0.99151525),  (0.99122925),  (0.99127075),  (0.99072925),  (0.9915365),  (0.99123575),  (0.99074425),  (0.99110425),  (0.9921675)]
hits_4_channels_Transformer =[ (0.98903525),  (0.98992225),  (0.986722),  (0.98931775),  (0.98930325),  (0.988786),  (0.98467725),  (0.989752),  (0.98888175),  (0.98165825)]

ttestPrint(hits_4_channels_PlusKernel_Transformer, hits_4_channels_Transformer)
#0.002944528275548179

small_ztf_Transformer =  [ (0.85118878),  (0.89060556),  (0.87974611),  (0.84909856),  (0.87502222),  (0.87970222),  (0.87623533),  (0.87922178),  (0.87876656),  (0.872923)]
small_ztf_PlusKernel_Transformer =  [ (0.90049778),  (0.91256678),  (0.89821078),  (0.91441789),  (0.91498033),  (0.91116667),  (0.90366922),  (0.91102111),  (0.91205667),  (0.90096844)]
ttestPrint(small_ztf_Transformer, small_ztf_PlusKernel_Transformer)
#4.034261360110064e_06

#%%

#TABLE III Transform Selection
#AUROC
hits_4_channels_Manual_35_2_Transform =  [(0.993201), (0.99161825), (0.99145025), (0.99296525), (0.9913495), (0.991835), (0.992156), (0.99183725), (0.99193), (0.9920725)]
hits_4_channels_Transformer =[ (0.98903525),  (0.98992225),  (0.986722),  (0.98931775),  (0.98930325),  (0.988786),  (0.98467725),  (0.989752),  (0.98888175),  (0.98165825)]

ttestPrint(hits_4_channels_Manual_35_2_Transform, hits_4_channels_Transformer)
#0.0006827220613235778

#acc
hits_4_channels_Manual_35_2_Transform_Acc =  [(0.97), (0.966), (0.9705), (0.96975), (0.9675), (0.96475), (0.96975), (0.96775), (0.96825), (0.964)]
hits_4_channels_Transformer_acc =  [(0.96825), (0.9685), (0.9665), (0.96725), (0.96775), (0.9665), (0.96225), (0.965), (0.96725), (0.95375)]
ttestPrint(hits_4_channels_Manual_35_2_Transform_Acc, hits_4_channels_Transformer_acc)
#0.13369887896734312

#AUROC
small_ztf_Transformer =  [ (0.85118878),  (0.89060556),  (0.87974611),  (0.84909856),  (0.87502222),  (0.87970222),  (0.87623533),  (0.87922178),  (0.87876656),  (0.872923)]
small_ztf_PlusKernel_Transformer_after_selection_29  =[(0.92467856), (0.90935733), (0.92589711), (0.91672922), (0.91298467), (0.90359922), (0.91629644), (0.91967278), (0.91783511), (0.90742467), (0.89729522), (0.92098756), (0.90957022), (0.91190589)]
ttestPrint(small_ztf_Transformer, small_ztf_PlusKernel_Transformer_after_selection_29)
#5.336269590679978e_07

#acc
small_ztf_Transformer_acc  = [(0.775), (0.81233333), (0.79616667), (0.79383333), (0.79233333), (0.79533333), (0.79183333), (0.80616667), (0.81666667), (0.79716667)]
small_ztf_PlusKernel_Transformer_after_selection_29_acc = [(0.85733333), (0.837), (0.8565), (0.84766667), (0.84816667), (0.83016667), (0.84783333), (0.8535), (0.849), (0.83633333), (0.8285), (0.84016667), (0.84516667), (0.83633333)]
ttestPrint(small_ztf_Transformer_acc, small_ztf_PlusKernel_Transformer_after_selection_29_acc)

#acc 97.73
small_ztf_Transformer_ac_97 =   [(0.75933333), (0.792), (0.7795), (0.77483333), (0.77616667), (0.781), (0.76616667), (0.78266667), (0.79), (0.7785)]
small_ztf_PlusKernel_Transformer_after_selection_29_ac_97 =   [(0.83), (0.82233333), (0.82716667), (0.83283333), (0.83066667), (0.82466667), (0.82933333), (0.83083333), (0.83516667), (0.8285), (0.82233333), (0.82733333), (0.825), (0.82783333)]
ttestPrint(small_ztf_Transformer_ac_97, small_ztf_PlusKernel_Transformer_after_selection_29_ac_97)


#acc97.7 hits
hits_4_channels_Transformer_ac_97 =   [(0.97175), (0.972), (0.97175), (0.97425), (0.97175), (0.9715), (0.9655), (0.9715), (0.972), (0.9545)]
[0.96965, 0.005468317840067445]
hits_4_channels_Manual_35_2_Transform_ac_97 =   [(0.97275), (0.96975), (0.97225), (0.97275), (0.97275), (0.9715), (0.97325), (0.97275), (0.97175), (0.97175)]
[0.9721249999999999, 0.0009568829604502237]
ttestPrint(hits_4_channels_Manual_35_2_Transform_ac_97, hits_4_channels_Transformer_ac_97)



#TABLE IV

#hits
hits_4_channels_Transformer_NO_EARLYSTOP =   [0.9863899999999999, 0.98160825, 0.98385925, 0.9884505, 0.9884715000000001, 0.9846090000000001, 0.98554375, 0.9874827499999999, 0.9843824999999999, 0.98940925]
hits_4_channels_Manual_35_2_Transform =  [(0.993201), (0.99161825), (0.99145025), (0.99296525), (0.9913495), (0.991835), (0.992156), (0.99183725), (0.99193), (0.9920725)]

print('Table IV HiTS')
ttestPrint(hits_4_channels_Transformer_NO_EARLYSTOP, hits_4_channels_Manual_35_2_Transform)

#ztf
small_ztf_Transformer_NO_EARLY_STOP =  [ (0.85118878),  (0.89060556),  (0.87974611),  (0.84909856),  (0.87502222),  (0.87970222),  (0.87623533),  (0.87922178),  (0.87876656),  (0.872923)]
small_ztf_PlusKernel_Transformer_after_selection_29  =[(0.92467856), (0.90935733), (0.92589711), (0.91672922), (0.91298467), (0.90359922), (0.91629644), (0.91967278), (0.91783511), (0.90742467), (0.89729522), (0.92098756), (0.90957022), (0.91190589)]
print('Table IV ZTF')
ttestPrint(small_ztf_Transformer_NO_EARLY_STOP, small_ztf_PlusKernel_Transformer_after_selection_29)