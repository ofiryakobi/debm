import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
class Model:
    def __init__(self,parameters, prospects,nsim,FullFeedback=True):
        """
        A Basic unit storing parameters, prospects and the number of simulations 
        required to make predictions.
        """
        if len(set([p._trials for p in prospects]))>1:
            raise Exception("Number of trials differ between prospects")
        self._trials_=prospects[0]._trials
        self._Num_of_prospects_=len(prospects) #number of prospects to model
        self.prospects=prospects #prospects to model
        self._pred_choices_=np.zeros((self._trials_,self._Num_of_prospects_))
        self._obs_choices_=None
        self._predictions_dict_={} # Dictionary with parameters and predictions        
        self.nsim=nsim
        self.parameters=parameters #Should be a dictionairy
        self.name=None
        self.FullFeedback=FullFeedback
    def set_obs_choices(self,oc): #Validate and save observed choices
        if callable(oc):
            self._obs_choices_=oc
        else:
            self._obs_choices_=oc.copy()
            oc=np.array(oc)
            if oc.shape!=self._pred_choices_.shape:
                raise Exception("Observed choices should have the same shape of the predictions - a column for each prospect, a row for each trial")
            self._obs_choices_=oc.copy()
    def get_obs_choices(self):
        if callable(self._obs_choices_):
            return(self._obs_choices_())
        else:
            return(self._obs_choices_)
    def get_predictions(self):
        return(self._pred_choices_)
    def save_predictions(self, fname):
        np.savetxt(fname, self._pred_choices_,'%10.5f',delimiter=',',newline='\n')
    def loss(self,loss="MSE",scope="prospectwise"):
        if loss.upper() in ["MSE","MSD"]:
            if scope.lower() in ["bitwise","bit","bw"]:
                return((np.square(self._pred_choices_ - self.get_obs_choices())).mean())
            elif scope.lower() in ["prospectwise","pw","prospect"]:
                return((np.square(self._pred_choices_.mean(axis=0) - self.get_obs_choices().mean(axis=0))).mean())
            else:
                raise Exception("Provide scope of loss calculation (bitwise, prospectwise")
        elif loss.upper()=="LL":
            self._epsilon_=0.0001
            obs=self.get_obs_choices()
            pred=np.clip(self._pred_choices_.copy(),self._epsilon_,1-self._epsilon_)
            if scope.lower() in ["bitwise","bit","bw"]:
                return(np.sum(obs*np.log(pred))*-1)
            elif scope.lower() in ["prospectwise","pw","prospect"]:
                return(np.sum(obs.mean(axis=0)*np.log(pred.mean(axis=0)))*-1)
            else:
                raise Exception("Provide scope of loss calculation (bitwise, prospectwise")
        else:
            raise Exception("Provide loss function [MSE/LL]")
    def _plot_(self,fig, axes,data,blocks=None,**args):
        colors=['green','black','red','blue','orange', 'purple']
        c=itertools.cycle(colors)
        if blocks==None:
            axes.plot(data,linewidth=args['linewidth'],dashes=args['dashes'])
            axes.set_title("Model type: "+ str(self.name))
            axes.set_xlabel("Trial")
            axes.set_ylabel("Choice rate")
            axes.set_xticks(range(self._trials_))
            axes.set_xticklabels(list(range(1,1+self._trials_)))
        else:
            axes.plot(np.mean(np.split(data,blocks),axis=1),linewidth=args['linewidth'],marker=args['marker'],dashes=args['dashes'])
            axes.set_title("Model type: "+ str(self.name))
            axes.set_xlabel("Block")
            axes.set_ylabel("Choice rate")
            axes.set_xticks(range(blocks))
            axes.set_xticklabels(list(range(1,1+blocks)))
        for line in axes.lines:
            line.set_color(next(c))
        axes.legend(list(map(chr,range(65,65+data.shape[1]+1))),title='Prospects')
        return(fig,axes)
    def plot_predicted(self,blocks=None,**args):
        fig, axes = plt.subplots(1,1)
        return(self._plot_(fig, axes,self._pred_choices_,blocks,linewidth=2,dashes=[1],marker='o',args=args))

    def plot_observed(self,blocks=None,**args):
        fig, axes = plt.subplots(1,1)
        return(self._plot_(fig, axes,self.get_obs_choices(),blocks,linewidth=2,marker='o',dashes=[],args=args))
    def plot_fit(self,blocks=None,**args):
        colors=['green','black','red','blue','orange', 'purple']
        fig, axes = plt.subplots(1,1)
        fig,axes=self._plot_(fig, axes,self.get_obs_choices(),blocks,linewidth=2,marker='o',dashes=[],args=args)
        fig,axes=self._plot_(fig, axes,self._pred_choices_,blocks,linewidth=2,dashes=[1],marker='o',args=args)        
        c=colors[0:self._Num_of_prospects_]*2
        legend_text=list(map(chr,range(65,65+len(c)//2)))*2
        for i,line in enumerate(axes.lines):
            line.set_color(c[i])
            if i>(len(c)//2-1):
                legend_text[i]='Predicted '+legend_text[i]
            else:
                legend_text[i]='Observed '+legend_text[i]
        axes.legend(legend_text,title='Prospects')        
        return(fig,axes)
    def CalcLoss(self,parameters,*args,**kwargs): #takes parameters, predict, calculate loss
        self.parameters=parameters
        try:
            reGenerate=kwargs['reGenerate']
        except:
            reGenerate=True            
        if len(args)==3: #reGenerated is passed! the length depends on the number of arguments in the loss function
            reGenerate=args[0]
            args=args[1:3]
        if args[0].upper() in ['AMLE','AML']: 
            if not callable(self._obs_choices_):
                raise Exception("For an aMLE to work, the observed choices of the model should be a function - for generating outcomes in every iteration")
            self._epsilon_=0.0001
            LLs=np.zeros(self.nsim)
            for s in range(self.nsim):
                obs=self.get_obs_choices()
                pred=np.clip(self.Predict(False),self._epsilon_,1-self._epsilon_)
                LLs[s]=np.sum(obs*np.log(pred))*-1
            return(np.mean(LLs))
        else:
            _=self.Predict(reGenerate)
            return self.loss(*args)
    def OptimizeBF(self,pars_dicts,pb=False,*args,**kwargs): #Brute force - try all in kappa_list
        if type(self._obs_choices_)==None:
            raise Exception("You have to store observations in the model first before fitting")
        minloss=9999
        bestp=None
        tmpm_array=[]
        for p in tqdm(pars_dicts,disable=not pb):
            tmpm=self.CalcLoss(p,*args,**kwargs)
            tmpm_array.append(tmpm)
            if tmpm<minloss:
                minloss=tmpm
                bestp=p
        res=dict(bestp=bestp,minloss=minloss,losses=tmpm_array,parameters_checked=pars_dicts)
        self.parameters=bestp
        return res

