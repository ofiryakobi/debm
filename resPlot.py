import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def resPlot(res):
    losses=res['losses']
    checked=res['parameters_checked']
    if len(checked[0])>2:
        print("More than two parameters - plotting iterations instead of parameters")
        xaxis=range(1,len(losses)+1)
        fig, axes = plt.subplots(1,1)
        axes.plot(xaxis,losses)
        axes.set_ylabel("Loss")
        axes.set_xlabel('Iteration')
        axes.axes.axhline(y=res['minloss'],color='r', linestyle='--')
        return fig,axes
    elif len(checked[0])==2:
        pnames=list(res['parameters_checked'][0].keys())
        xaxis=np.unique([x[pnames[0]] for x in checked])
        yaxis=np.unique([x[pnames[1]] for x in checked])
        X,Y=np.meshgrid(xaxis,yaxis)
        Z=np.reshape(losses,(len(xaxis),len(yaxis)))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5)
        ax.scatter(res['bestp'][pnames[0]],res['bestp'][pnames[1]],res['minloss'],color='black')
        ax.set_xlabel(pnames[0])
        ax.set_ylabel(pnames[1])
        ax.set_zlabel("Loss")
        ax.set_title("3D visualization of the loss function")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return fig, ax, surf
    else:
        pname=list(res['parameters_checked'][0].keys())[0]
        xaxis=np.array([x[pname] for x in checked])
        fig, axes = plt.subplots(1,1)
        axes.plot(xaxis,losses)
        axes.set_ylabel("Loss")
        axes.set_xlabel(pname)
        axes.set_title("Visualization of the loss function")
        axes.axes.axvline(x=res['bestp'][pname],color='r', linestyle='--')
        return fig,axes