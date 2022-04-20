import os
import configs

args = configs.get_args()
stat_path = args.stat_path + "stats_"+args.dataset+"_"+args.arch+"/stats_"+args.stat_layer+"_"+args.data_type+"_splz"+str(args.stat_bz)+"/"+args.hook_type
n = int(len(os.listdir(stat_path))/2)
print(n-1)