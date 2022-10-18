from three_fft_initial_sampling_msm_def import predict


sampling_num = [10,15,20]
ranges = [[5,15],[5,20],[10,15],[10,20]]
# start = [0.5,1.0,1.5,5.0]
# end = [1.5,]
dic_dim = [75,100,125]
in_dim = [75,100,125]

for sam_num in sampling_num:
    for r in ranges:
        for d in dic_dim:
            for i in in_dim:
                start = r[0]
                end = r[1]
                predict(sam_num,start,end,d,i)
