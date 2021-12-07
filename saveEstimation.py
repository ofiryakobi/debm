import pandas as pd
def saveEstimation(res,fname,**kwargs):
    """
    saveEstimation(res,fname,**kwargs)
    res is a results dictionary created by the model.
    Save estimation results to a csv file. Supply file name (including path, if needed) and other optional keyword arguments
    """
    df=pd.DataFrame()
    ind=list(range(len(res['losses'])))
    df=pd.concat([pd.DataFrame(res['bestp'],index=ind),pd.DataFrame({'loss':res['losses']})],axis=1)
    df.to_csv(fname,index=False,**kwargs)
    return df