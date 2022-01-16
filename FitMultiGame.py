import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
class FitMultiGame(): #All models should be of the same type
    def __init__(self,models,obs_choices):
        if len(models)!=len(obs_choices):
            raise Exception("Number of models and observed choices array differ")
        self._models=models
        self._obs_choices=obs_choices
        self._mp=None
    def CalcLoss(self,par, *args, **kwargs):
        losses=np.zeros(len(self._models))
        for i,m in enumerate(self._models):
            m.set_obs_choices(self._obs_choices[i])
            l=m.CalcLoss(par,*args, **kwargs)
            losses[i]=l
        if len(args)==3: #reGenerate is passed. the length depends on the number of arguments in the loss function
            loss=args[1]
        elif len(args)==2:
            loss=args[0]
        else:
            raise Exception('Wrong number of arguments: '+str(args))
        if loss.upper()=="MSE":
            return(np.mean(losses))
        elif loss.upper()=="LL":
            return(np.sum(losses))        
    def OptimizeBF(self,pars_dicts,pb=False,*args,**kwargs): #Brute force - try all in kappa_list
        if self._mp==None:
            return self.sp_OptimizeBF(pars_dicts,pb,*args,**kwargs)
        else:
            return self.mp_OptimizeBF(pars_dicts,pb,*args,**kwargs)
    def sp_OptimizeBF(self,pars_dicts,pb=False,*args,**kwargs): #Brute force - try all in list
        """
        pars_dicts should be a list of dicts. e.g., [{'alpha':2},{'alpha':4}]
        """
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
        return res
    def mp_OptimizeBF(self,pars_dicts,pb=False,*args,**kwargs): #Brute force - try all in kappa_list
        minloss=9999
        bestp=None
        if pb:
            with mp.Pool(self._mp) as pool:
                tmpm_array=list(tqdm(pool.imap(partial(_mpoptimizebf_,self,*args,**kwargs),pars_dicts),total=len(pars_dicts)))
        else:
            with mp.Pool(self._mp) as pool:
                tmpm_array=pool.map(partial(_mpoptimizebf_,self,*args,**kwargs),pars_dicts)
            
        minloss=np.min(tmpm_array)
        bestp=pars_dicts[np.argmin(tmpm_array)]
        res=dict(bestp=bestp,minloss=minloss,losses=tmpm_array,parameters_checked=pars_dicts)
        self.parameters=bestp
        return res

    @property
    def mp(self):
        return self._mp
    @mp.setter
    def mp(self, new_value):
        if type(new_value)==int:
            if new_value>1:
                self._mp = np.min([new_value,mp.cpu_count()])
                print("Multiprocessing is now on for this multigame.")
                print("Your computer has {} available CPUs. You chose to use {} CPUs.".format(mp.cpu_count(),self._mp))
            else:
                self._mp=None
                print("Your computer has {} available CPUs.".format(mp.cpu_count()))
                print("Multiprocessing is now disabled for this multigame")
        else:
            self._mp=None
            print("Your computer has {} available CPUs.".format(mp.cpu_count()))
            print("None numeric input. Multiprocessing is now disabled for this multigame")
def _mpoptimizebf_(self,*args,**kwargs): # Helper function for multiprocessing
    p=args[-1]
    args=args[:-1]
    return self.CalcLoss(p,*args,**kwargs)
