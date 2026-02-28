import numpy as np
from scipy.stats import ttest_rel
import scipy.stats as stats
from scipy.stats import friedmanchisquare

ei_auc = np.array([
0.578125,
0.5859375,
0.5,
0.5625,
0.5859375,
0.6328125,
0.453125,
0.4453125,
0.625,
0.6875,
0.6875,
0.75,
0.578125,
0.828125,
0.6328125,
0.828125,
0.5859375,
0.6796875,
0.75,
0.8515625
])

pi_auc = np.array([
0.578125,
0.5859375,
0.5,
0.53125,
0.5859375,
0.4140625,
0.6796875,
0.6875,
0.625,
0.6875,
0.6328125,
0.5625,
0.578125,
0.8671875,
0.8828125,
0.734375,
0.7109375,
0.4765625,
0.84375,
0.6328125
])

lcb_auc = np.array([
0.578125,
0.5859375,
0.5859375,
0.71875,
0.7890625,
0.6015625,
0.8828125,
0.5625,
0.8671875,
0.734375,
0.875,
0.71875,
0.8671875,
0.78125,
0.625,
0.6328125,  
0.453125,
0.8671875,
0.6875,
0.4140625])

# baseline_auc = np.array([
#     0.6094,
#     0.6484,
#     0.4922,
#     0.4688,
#     0.4375,
#     0.6562,
#     0.4141,
#     0.3984,
#     0.6328,
#     0.8047,
#     0.5781,
#     0.6562,
#     0.4062,
#     0.5234,
#     0.4297,
#     0.5078,
#     0.4609,
#     0.6172,
#     0.4141
# ])

base_auc = np.array([0.609375, 
                     0.6484375, 
                     0.4921875, 
                     0.46875, 
                     0.4375, 
                     0.65625, 
                     0.4140625, 
                     0.3984375, 
                     0.6328125, 
                     0.8046875, 
                     0.578125, 
                     0.65625, 
                     0.40625, 
                     0.5234375, 
                     0.4296875, 
                     0.5078125, 
                     0.4609375, 
                     0.6171875, 
                     0.4140625, 
                     0.578125] )

# Paired t-test

base_ei_diff = base_auc - ei_auc
base_pi_diff = base_auc - pi_auc
base_lcb_diff = base_auc - lcb_auc

# make a "one sample t-test" for the differences (this time we only use the built-in python function):
test1 = stats.ttest_1samp(base_ei_diff, popmean=0)
test2 = stats.ttest_1samp(base_pi_diff, popmean=0)
test3 = stats.ttest_1samp(base_lcb_diff, popmean=0)
print('Base vs EI:', test1[1]) # p-value for the null hypothesis that the mean difference is 0
print('Base vs PI:', test2[1]) # p-value for the null hypothesis that the mean difference is 0
print('Base vs LCB:', test3[1]) # p-value for the null hypothesis that the mean difference is 0


# Comparing the three methods together using Friedman test
statistic, p_value = friedmanchisquare(ei_auc, pi_auc, lcb_auc)
print('Friedman test statistic:', statistic)
print('Friedman test p-value:', p_value)