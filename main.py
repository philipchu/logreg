# This is a sample Python script.
import log_reg_ll
import log_reg_becker
import numpy as np
import timeit

simulated_separableish_features, simulated_labels = log_reg_becker.gen_data()
beck_rt = timeit.default_timer()
log_reg_becker.run_preds(simulated_separableish_features, simulated_labels, num_steps=100)
beck_rt = timeit.default_timer() - beck_rt
print(beck_rt)

pc_rt = timeit.default_timer()
log_reg_ll.run_preds(np.transpose(np.matrix(simulated_separableish_features)),
                     np.transpose(np.matrix(simulated_labels)), num_steps=1)
pc_rt = timeit.default_timer() - pc_rt
print(pc_rt)
