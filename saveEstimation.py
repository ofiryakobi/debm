import pandas as pd
def saveEstimation(res,fname,**kwargs):
    """
    Save estimation results to a csv file. Supply file name (including path, if needed) and other optional keyword arguments
    """
    df=pd.concat([pd.DataFrame(res['bestp']),pd.DataFrame({'loss':res['losses']})],axis=1)
    df.to_csv(fname,index=False,**kwargs)
    return df