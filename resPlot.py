import numpy as np
import matplotlib.pyplot as plt
def resPlot(res):
    losses=res['losses']
    checked=res['parameters_checked']
    if len(checked[0])>1:
        raise Warning("More than one parameter - plotting iterations instead of parameters")
        xaxis=range(1,len(losses))
        fig, axes = plt.subplots(1,1)
        axes.plot(xaxis,losses)
        axes.set_ylabel("Loss")
        axes.set_xlabel('Iteration')
        axes.axes.axhline(y=res['minloss'],color='r', linestyle='--')
        return fig,axes
    else:
        pname=list(res['parameters_checked'][0].keys())[0]
        xaxis=np.array([x[pname] for x in checked])
        fig, axes = plt.subplots(1,1)
        axes.plot(xaxis,losses)
        axes.set_ylabel("Loss")
        axes.set_xlabel(pname)
        axes.axes.axvline(x=res['bestp'][pname],color='r', linestyle='--')
        return fig,axes