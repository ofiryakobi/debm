##
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize
#from multiprocessing import Pool


class Prospect:
    """
    This is a prospect in a decision-making problem.
    Define the number of trials (round) agents are faced in this problem,
    fun is the function that generates these outcomes (in case these are stochastic outcomes),
    you can send an array in the size of trials instead of a function.
    
    In case the function you enter generates one outcomes (one value) per run, 
    oneOutcomePerRun should be True. Else, for example if the function returns n values (where n is the number of trials),
    it should be set to False.
    *args and **kwargs are parameters passed to this function.
    
    For example, if you want to use Numpy (you have to import it in your code) to pick random choices,
    either 50 or -1000 in probability of 90% and 10% respectiveley, fun should be np.random.choice
    and the arguments should be set accordingly. For example, the following command:
    A=Prospect(trials,np.random.choice,False,[3,0],100,True,[0.45,0.55])
    
    A produces 3 (in probability of 45%) or 0 otherwise, using NumPy. The function was asked
    to return 100 values at once (notice the False after the function name), with replacement (hence the True after [3,0]).
    
    """
    def __init__(self,trials:int,fun,oneOutcomePerRun:bool,*args,**kwargs):
        if type(trials)!=int:
            raise Exception("Trials should be an integer (e.g., 100)")
        elif trials<1:
            raise Exception("There should be at least one trial")
        self._trials=trials #How many trials (rows)
        self.dtype=np.float64
        self._PopFun=fun
        self._PopPars=args,kwargs
        self._oneOutcomePerRun=oneOutcomePerRun
        self.Generate()
    def Generate(self): #put values in outcomes. Check if user passed function or numbers
        """
        Re-generate outcome values if fun is a function
        """
        self.outcomes=np.zeros(self._trials,dtype=self.dtype)
        if callable(self._PopFun):
            if self._oneOutcomePerRun:
                for i in range(self._trials):
                    self.outcomes[i]=self._PopFun(*self._PopPars[0],**self._PopPars[1])
            else:
                self.outcomes=self._PopFun(*self._PopPars[0],**self._PopPars[1])
        else:
            self.outcomes=np.array(self._PopFun)
        self.outcomes=self.outcomes.astype(self.dtype)
    def EV(self,EVsims=10000):
        return(np.mean([self._PopFun(*self._PopPars[0],**self._PopPars[1]) for i in range(EVsims)]))
    def __eq__(self,other):
        if self._trials!=other._trials:
            raise Exception("The number of trials is not the same for the two prospects")
        n=1000
        gt=np.zeros(n)
        for i in range(n):
            self.Generate()
            gt[i]=100*np.sum(self.outcomes==other.outcomes)/self._trials
        print(f'The first prospect equals the second one in {gt.mean()}% of the trials.')
    def __gt__(self,other):
        if self._trials!=other._trials:
            raise Exception("The number of trials is not the same for the two prospects")
        n=1000
        gt=np.zeros(n)
        for i in range(n):
            self.Generate()
            gt[i]=100*np.sum(self.outcomes>other.outcomes)/self._trials
        print(f'The first prospect is better than the second one in {gt.mean()}% of the trials.')
    def __lt__(self,other):
        if self._trials!=other._trials:
            raise Exception("The number of trials is not the same for the two prospects")
        n=1000
        gt=np.zeros(n)
        for i in range(n):
            self.Generate()
            gt[i]=100*np.sum(self.outcomes<other.outcomes)/self._trials
        print(f'The first prospect is worse than the second one in {gt.mean()}% of the trials.')


class Model:
    def __init__(self,parameters, prospects,nsim,FullFeedback=True):
        """
        Model's parameters: 
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

        if args[0].upper() in ['AMLE','AML']: #INCOREECT ALGORITHM
            if not callable(self._obs_choices_):
                raise Exception("For an aMLE to work, the observed choices of the model should be a function - for generating outcomes in every iteration")
            self._epsilon_=0.0001
            LLs=np.zeros(self.nsim)
            for s in range(self.nsim):
                #raise Exception("debug")
                obs=self.get_obs_choices()
                pred=np.clip(self.Predict(False),self._epsilon_,1-self._epsilon_)
                LLs[s]=np.sum(obs*np.log(pred))*-1
            return(np.mean(LLs))
        else:
            _=self.Predict(reGenerate)
            return(self.loss(*args))
    def OptimizeBF(self,pars_dicts,pb=False,*args,**kwargs): #Brute force - try all in kappa_list
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
    def Simplex(self,method,*args): #Brute force - try all in kappa_list
        res=minimize(fun=self.CalcLoss,x0=[1],args=args,method=method,options=dict(maxiter=100000))
        return(res)

class NaiveSampler(Model):
    def __init__(self,kappa_or_less,*args): #Kappa_or_less should be true if using the same as Yakobi et al. 2020
        Model.__init__(self,*args)
        self._K=self.parameters['Kappa']
        self._Kappa_or_less=kappa_or_less
        self.name="NaiveSampler"
    def Predict(self,reGenerate=True):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
        elif type(self.parameters)==list:
            self._K=int(self.parameters[0])
        else:
            self._K=int(self.parameters)
        if self._K in self._predictions_dict_ and False: # DISABLED Re-use predictions
            self._pred_choices_=self._predictions_dict_[self._K]
            return(self._predictions_dict_[self._K])
        else:
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                K=np.random.randint(1,self._K+1)# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
                for p in self.prospects:
                    if callable(p._PopFun) and reGenerate:
                        p.Generate()
                choices=np.zeros(self._trials_,dtype=np.int16())
                if self._Kappa_or_less: #A naive-sampler model in which the size of the sample is Kappa or less, drawn in each trial
                    indices=[np.random.choice(i+1,size=np.random.choice(K)+1) for i in range(self._trials_-1)]  #samples to draw from
                else:
                    indices=[np.random.choice(i+1,size=K) for i in range(self._trials_-1)]  #samples to draw from
                data=np.vstack([x.outcomes for x in self.prospects]).transpose()
                means=np.zeros((data.shape[0]-1,data.shape[1]),dtype=np.float16)
                choices[0]=np.random.randint(0,self._Num_of_prospects_)
                grand_choices[s,0,choices[0]]=1
                for i,idx in enumerate(indices):
                    data[idx].mean(axis=0,out=means[i])
                    choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                    grand_choices[s,i+1,choices[i+1]]=1
            self._pred_choices_=np.mean(grand_choices,axis=0)
            self._predictions_dict_[self._K]=self._pred_choices_
            return(self._pred_choices_)

        
class FullData(Model):
    def __init__(self,*args): #Kappa_or_less should be true if using the same as Yakobi et al. 2020
        Model.__init__(self,*args)
        self.name="Full-data (Fictitious play)"
    def Predict(self):
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            for p in self.prospects:
                if callable(p.PopFun):
                    p.Generate()
            choices=np.zeros(self._trials_,dtype=np.int16())
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            means=np.zeros((data.shape[0]-1,data.shape[1]),dtype=np.float16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            for i in range(self._trials_-1):
                data[0:i+1].mean(axis=0,out=means[i])
                choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                grand_choices[s,i+1,choices[i+1]]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return(self._pred_choices_)


class Population_1or99(Model):
    def __init__(self,*args): #Kappa_or_less should be true if using the same as Yakobi et al. 2020
        Model.__init__(self,*args)
        self.name="Population 1 or 99"
    def Predict(self):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            K=np.random.choice([1,99])# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
            for p in self.prospects:
                if callable(p.PopFun):
                    p.Generate()
            choices=np.zeros(self._trials_,dtype=np.int16())
            indices=[np.random.choice(i+1,size=min(K,i+1),replace=False) for i in range(self._trials_-1)]  #samples to draw from
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            means=np.zeros((data.shape[0]-1,data.shape[1]),dtype=np.float16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            for i,idx in enumerate(indices):
                data[idx].mean(axis=0,out=means[i])
                choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                grand_choices[s,i+1,choices[i+1]]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return(self._pred_choices_)

class Population_1or3f(Model):
    def __init__(self,*args): 
        Model.__init__(self,*args)
        self.name="Population first 1 or 3"
    def Predict(self):
        print("predict pop1or99_1")
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            K=np.random.choice([1,3])# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
            for p in self.prospects:
                if callable(p.PopFun):
                    p.Generate()
            choices=np.zeros(self._trials_,dtype=np.int16())
            indices=[list(range(min(i+1,K))) for i in range(self._trials_-1)]  #samples to draw from
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            means=np.zeros((data.shape[0]-1,data.shape[1]),dtype=np.float16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            for i,idx in enumerate(indices):
                data[idx].mean(axis=0,out=means[i])
                choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                grand_choices[s,i+1,choices[i+1]]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return(self._pred_choices_)


class Sample_of_K(Model):
    def __init__(self,*args): #Kappa_or_less should be true if using the same as Yakobi et al. 2020
        Model.__init__(self,*args)
        self._K=self.parameters['Kappa']
        self.name="Sample of K"
    def Predict(self, reGenerate):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
        elif type(self.parameters)==list:
            self._K=int(self.parameters[0])
        else:
            self._K=int(self.parameters)
        if self._K in self._predictions_dict_ and False: # DISABLED Re-use predictions
            self._pred_choices_=self._predictions_dict_[self._K]
            return(self._predictions_dict_[self._K])
        else:
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                K=self._K
                if reGenerate:
                    for p in self.prospects:
                        if callable(p.PopFun):
                            p.Generate()
                choices=np.zeros(self._trials_,dtype=np.int16())
                data=np.vstack([x.outcomes for x in self.prospects]).transpose()
                means=np.zeros((data.shape[0]-1,data.shape[1]),dtype=np.float16)
                choices[0]=np.random.randint(0,self._Num_of_prospects_)
                grand_choices[s,0,choices[0]]=1
                for i in range(self._trials_-1):
                    idx=np.random.choice(i+1,size=K)
                    means[i,]=data[idx,].mean(axis=0)
                    choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                    grand_choices[s,i+1,choices[i+1]]=1
            self._pred_choices_=np.mean(grand_choices,axis=0)
            self._predictions_dict_[self._K]=self._pred_choices_
            return(self._pred_choices_)


class SAW(Model):
    def __init__(self,*args): #Kappa_or_less should be true if using the same as Yakobi et al. 2020
        Model.__init__(self,*args)
        self._K=self.parameters['Kappa']
        self._W=self.parameters['Omega']
        self._D=self.parameters['Delta']
        self.name="SAW"
    def Predict(self, reGenerate):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
            self._W=self.parameters['Omega']
            self._D=self.parameters['Delta']
        elif type(self.parameters)==list:
            self._K=int(self.parameters[0])
            self._W=self.parameters[1]
            self._D=self.parameters[2]
        else:
            raise Exception("Parameters should be entered correctly as a dictionairy or list (Kappa,Omega,Delta)")
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            K=np.random.randint(1,self._K+1)
            W=np.random.uniform(0,self._W)
            D=np.random.uniform(0,self._D)
            if reGenerate:
                for p in self.prospects:
                    if callable(p.PopFun):
                        p.Generate()
            choices=np.zeros(self._trials_,dtype=np.int16())
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            means=np.full((data.shape[0]-1,data.shape[1]),-np.inf,dtype=np.float16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            ws=np.random.choice(a=[0,0.5],p=[1-W,W],size=self._trials_-1)
            for i in range(self._trials_-1):
                consider=np.random.uniform(size=self._Num_of_prospects_)>D**((i+1)/(i+2))
                if np.sum(consider)<1:
                    consider[np.random.randint(consider.shape[0])]=True
                consider=np.where(consider)[0]
                idx=np.random.choice(i+1,size=K)
                means[i,consider]=(1-ws[i])*data[idx][:,consider].mean(axis=0)+ws[i]*data[0:i+1,consider].mean(axis=0)
                choices[i+1]=np.random.choice(np.argwhere(means[i,:]==np.amax(means[i,:])).flatten())
                grand_choices[s,i+1,choices[i+1]]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        self._predictions_dict_[self._K]=self._pred_choices_
        return(self._pred_choices_)


class NaiveSampler_2S(Model):
    def __init__(self,risky_prospects,*args): #Used in Yakobi et al. 2020, risky_prospects=list of prospect indices starting from 0
        Model.__init__(self,*args)
        self.name="2-stage NaiveSampler"
        self._risky_prospects_=risky_prospects
        self._safe_prospects_=list(set(range(self._Num_of_prospects_))-set(self._risky_prospects_))
    def Predict(self, reGenerate):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
        elif type(self.parameters)==list:
            self._K=int(self.parameters[0])
        else:
            self._K=int(self.parameters)
        if self._K in self._predictions_dict_: # Re-use predictions
            return(self._predictions_dict_[self._K])
        else:
            rp=np.array(self._risky_prospects_,dtype=np.int16)
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                K=np.random.randint(1,self._K+1)# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
                for p in self.prospects:
                    if callable(p.PopFun):
                        p.Generate()
                choices=np.zeros(self._trials_,dtype=np.int16)
                indices1=[np.random.choice(i+1,size=K) for i in range(self._trials_-1)]  #samples to draw from 1st stage
                indices2=[np.random.choice(i+1,size=K) for i in range(self._trials_-1)]  #samples to draw from 2nd stage
                data=np.vstack([x.outcomes for x in self.prospects]).transpose()
                means1=np.zeros((data.shape[0]-1,len(rp)),dtype=np.float16) #First stage
                means2=np.zeros((data.shape[0]-1,data.shape[1]-(len(rp)-1)),dtype=np.float16) #2nd stage, Number of columns is reduced by len(rp)-1 because only one is chosen in the 1st stage
                choices[0]=np.random.randint(0,self._Num_of_prospects_)
                grand_choices[s,0,choices[0]]=1
                for i,idx in enumerate(indices1):
                    data[idx,:][:,rp].mean(axis=0,out=means1[i,]) # Only the risky choices
                    choice_1stage=np.array(rp[np.random.choice(np.argwhere(means1[i,:]==np.amax(means1[i,:])).flatten())])
                    #2nd stage
                    prospects_for_2ndstage=np.array(list((set(range(self._Num_of_prospects_))-set(rp))|set(choice_1stage.flatten()))) #These are the finalists: the safe option/s + chosen risky one
                    means2[i,np.where(prospects_for_2ndstage==self._safe_prospects_)[0]]=data[indices2[i]][:,self._safe_prospects_].mean(axis=0)
                    indices2[i]=[*indices2[i],*idx] # Add 1st stage samples for the calculation of the risky options (those chosen in stage 1)                    
                    means2[i,np.where(prospects_for_2ndstage==choice_1stage)[0]]=data[indices2[i]][:,choice_1stage].mean(axis=0) # Calcualate mean of the risky options
                    choices[i+1]=prospects_for_2ndstage[np.random.choice(np.argwhere(means2[i,:]==np.amax(means2[i,:])).flatten())]
                    grand_choices[s,i+1,choices[i+1]]=1
            self._pred_choices_=np.mean(grand_choices,axis=0)
            self._predictions_dict_[self._K]=self._pred_choices_
            return(self._pred_choices_)

class RL_delta_rule(Model):
    def __init__(self,choice_rule="best",*args): #Used in Yakobi et al. 2020, risky_prospects=list of prospect indices starting from 0
        Model.__init__(self,*args)
        self.name="Reinforcement Learning with Delta-rule"
        self._A=int(self.parameters['Alpha'])
        if choice_rule.lower() not in ['best','e-greedy','egreedy','e_greedy']:
            raise Exception("Please provide a choice rule (best or e-greedy)")
        self._choice_rule_=choice_rule
    def Predict(self, reGenerate):
        if not self.FullFeedback:
            raise Exception("This model does not currently work with partial-feedback data")
        if type(self.parameters)==dict:
            self._A=self.parameters['Alpha']
        elif type(self.parameters) in [list, np.ndarray]:
            self._A=self.parameters[0]
        else:
            self._A=self.parameters
        if self._A<0 or  self._A>1: 
            self._pred_choices_=np.full_like(self._pred_choices_, np.inf)
            return None
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            if reGenerate:
                for p in self.prospects:
                    if callable(p._PopFun):
                        p.Generate()
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            Qs=np.zeros_like(data)
            choices=np.zeros(self._trials_,dtype=np.int16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            for i in range(1,self._trials_):
                Qs[i]=Qs[i-1]*(1-self._A)+data[i-1]*self._A
                choices[i]=np.random.choice(np.argwhere(Qs[i,:]==np.amax(Qs[i,:])).flatten())
                grand_choices[s,i,choices[i]]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return(self._pred_choices_)
        
        

class ISAW2(Model):
    def __init__(self,risky_prospects,*args): #Used in Yakobi et al. 2020, risky_prospects=list of prospect indices starting from 0
        Model.__init__(self,*args)
        self.name="I-SAW2"
        self._risky_prospects_=risky_prospects
        self._safe_prospects_=list(set(range(self._Num_of_prospects_))-set(self._risky_prospects_))
    def Predict(self):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
            self._Nu=self.parameters['Nu']
            self._Omega=self.parameters['Omega']
        elif type(self.parameters)==list:
            self._K=int(self.parameters[0])
            self._Nu=self.parameters[1]
            self._Omega=self.parameters[2]
        else:
            raise Exception("Parameters should be a list (Kappa, Nu, Omega) or a dictionairy")
        tmpkey=str(self._K)+str(self._Nu)+str(self._Omega)
        if tmpkey in self._predictions_dict_: # Re-use predictions
            return(self._predictions_dict_[tmpkey])
        else:
            rp=np.array(self._risky_prospects_,dtype=np.int16)
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                K=np.random.randint(1,self._K+1)# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
                N=np.random.uniform(0,self._Nu)# New Nu for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
                W=np.random.uniform(0,self._Omega)# New Omega for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
                for p in self.prospects:
                    if callable(p.PopFun):
                        p.Generate()
                choices=np.zeros(self._trials_,dtype=np.int16)
                data=np.vstack([x.outcomes for x in self.prospects]).transpose()
                means1=np.zeros((data.shape[0]-1,len(rp)),dtype=np.float16) #First stage
                means2=np.zeros((data.shape[0]-1,data.shape[1]-(len(rp)-1)),dtype=np.float16) #2nd stage, Number of columns is reduced by len(rp)-1 because only one is chosen in the 1st stage
                means_final=np.zeros((data.shape[0]-1,data.shape[1]-(len(rp)-1)),dtype=np.float16)
                choices[0]=np.random.randint(0,self._Num_of_prospects_)
                grand_choices[s,0,choices[0]]=1
                for i in range(self._trials_-1):
                    if np.random.rand()<N: # Are we making a new choice in this trial?
                        idx1=np.random.choice(i+1,size=K)
                        data[idx1,:][:,rp].mean(axis=0,out=means1[i,]) # Only the risky choices
                        choice_1stage=np.array(rp[np.random.choice(np.argwhere(means1[i,:]==np.amax(means1[i,:])).flatten())])
                        #2nd stage
                        prospects_for_2ndstage=np.array(list((set(range(self._Num_of_prospects_))-set(rp))|set(choice_1stage.flatten()))) #These are the finalists: the safe option/s + chosen risky one
                        idx2=np.random.choice(i+1,size=K)
                        means2[i,np.where(prospects_for_2ndstage==self._safe_prospects_)[0]]=data[idx2][:,self._safe_prospects_].mean(axis=0)
                        idx3=np.random.choice(i+1,size=K)
                        means2[i,np.where(prospects_for_2ndstage==choice_1stage)[0]]=data[np.concatenate([idx3,idx2])][:,choice_1stage].mean(axis=0) # Calcualate mean of the risky options
                        means_final[i,np.where(prospects_for_2ndstage==prospects_for_2ndstage)[0]]=(1-W)*means2[i,np.where(prospects_for_2ndstage==prospects_for_2ndstage)[0]]+W*data[0:i+1,np.where(prospects_for_2ndstage==prospects_for_2ndstage)[0]].mean(axis=0)
                        choices[i+1]=prospects_for_2ndstage[np.random.choice(np.argwhere(means_final[i,:]==np.amax(means_final[i,:])).flatten())]
                    else:
                        choices[i+1]=choices[i] # Inertia - repeat last choice
                    grand_choices[s,i+1,choices[i+1]]=1
            self._pred_choices_=np.mean(grand_choices,axis=0)
            self._predictions_dict_[tmpkey]=self._pred_choices_
            return(self._pred_choices_)

class FitMultiGame(): #All models should be of the same type
    def __init__(self,models,obs_choices):
        if len(models)!=len(obs_choices):
            raise Exception("Number of models and observed choices array differ")
        self._models=models
        self._obs_choices=obs_choices
    def CalcLoss(self,par,loss,scope):
        losses=np.zeros(len(self._models))
        for i,m in enumerate(self._models):
            m.set_obs_choices(self._obs_choices[i])
            l=m.CalcLoss(par,loss,scope)
            losses[i]=l
        if loss.upper()=="MSE":
            return(np.mean(losses))
        elif loss.upper()=="LL":
            return(np.sum(losses))        
    def OptimizeBF(self,pars_dicts,pb=False,*args): #Brute force - try all in list
        """
        pars_dicts should be a list of dicts. e.g., [{'alpha':2},{'alpha':4}]
        """
        minloss=9999
        bestp=None
        tmpm_array=[]
        for p in tqdm(pars_dicts,disable=not pb):
            tmpm=self.CalcLoss(p,*args)
            tmpm_array.append(tmpm)
            if tmpm<minloss:
                minloss=tmpm
                bestp=p
        res=dict(bestp=bestp,minloss=minloss,losses=tmpm_array,parameters_checked=pars_dicts)                
        return res

def makeGrid(pars_dict):  # Make a grid suitable for Grid Search
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    return [dict(zip(keys,cc)) for cc in combinations]
    
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