from bqtmizer import bqtmizer

weights_dict = {"time":1/3, "cost":1/3, "num":1/3}
result = bqtmizer("../dataset/PaintControl_TCM.csv", ["rate"], ["time"], "xxx", weights=weights_dict, N=30, beta=0.9)