import numpy as np
import pandas as pd

def calculateReturns(prices, lag):
# =============================================================================
# prices is a pandasDataFrame type
# =============================================================================
    prevPrices = prices.shift(lag)
    return ((prices - prevPrices) / prevPrices)

def calculateMaxDD(cumret):
# =============================================================================
# calculation of maximum drawdown and maximum drawdown duration based on
# cumulative COMPOUNDED returns. cumret must be a compounded cumulative return.
# i is the index of the day with maxDD.
# Source: https://github.com/burakbayramli/books/blob/master/Quantitative_Trading
# _Chan/python/calculateMaxDD.py
# =============================================================================
    highwatermark = np.zeros(cumret.shape)
    drawdown = np.zeros(cumret.shape)
    drawdownduration = np.zeros(cumret.shape)

    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=(1+cumret[t])/(1+highwatermark[t])-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1

    maxDD, i = np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
    maxDDD = np.max(drawdownduration)
    return maxDD, maxDDD, i

def VisualizeTreePDF(treeModel, filename):
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    export_graphviz(treeModel, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True,
                    precision = 5)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_pdf(filename + ".pdf")
