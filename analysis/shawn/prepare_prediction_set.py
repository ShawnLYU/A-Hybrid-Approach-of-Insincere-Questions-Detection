import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="increase output verbosity")
parser.add_argument("output", help="increase output verbosity")

args = parser.parse_args()



filepath_train = args.input
df_train = pd.read_csv(filepath_train,index_col=0)

np.savetxt(args.output,df_train.question_text.values,fmt='%s')