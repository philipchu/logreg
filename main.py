# This is a sample Python script.
import log_reg_ll
import log_reg_becker
import numpy as np
import timeit

count = 0
while True:
    simulated_separableish_features, simulated_labels = log_reg_becker.gen_data()
    beck_rt = timeit.default_timer()
    _, beck_acc = log_reg_becker.run_preds(simulated_separableish_features, simulated_labels, num_steps=1500)
    beck_rt = timeit.default_timer() - beck_rt
    print(beck_rt)

    pc_rt = timeit.default_timer()
    _, pc_acc = log_reg_ll.run_preds(np.transpose(np.matrix(simulated_separableish_features)),
                         np.transpose(np.matrix(simulated_labels)), num_steps=1, num_batches=4) #why does batching only work for odds??
    pc_rt = timeit.default_timer() - pc_rt
    print(pc_rt)
    count += 1
    print(count)
    # if beck_acc > pc_acc:
    break
