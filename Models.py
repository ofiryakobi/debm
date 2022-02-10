import numpy as np
from .Model import Model
class NaiveSampler(Model):
    def __init__(self,kappa_or_less,*args): #Kappa_or_less should be true if using the same model as Yakobi et al. 2020
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
                K=self._K
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
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            K=np.random.choice([1,3])# New Kappa for each simulation (e.g., each agent) drawn from a uni distribution of [1-K]
            for p in self.prospects:
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
    def Predict(self, reGenerate=True):
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


class PAS(Model):
    def __init__(self,*args):
        Model.__init__(self,*args)
        self._K=self.parameters['Kappa']
        self._W=self.parameters['Omega']
        self._D=self.parameters['Delta']
        self.name="PAS"
    def Optimize_Simplex(self, parameters,*args,**kwargs):
        """
        Does not change Kappa (int), pass only Omega and Delta
        """
        if type(self.parameters) in [list,np.ndarray]:
            return self.CalcLoss(np.concatenate([[self.parameters[0]],parameters]), *args,**kwargs)
        elif type(self.parameters)==dict:
            return self.CalcLoss(np.concatenate([[self.parameters['Kappa']],parameters]), *args,**kwargs)
        else:
            raise Exception("Parameters should be entered correctly as a dictionairy or list (Kappa,Omega,Delta)")
    def Predict(self, reGenerate=True):
        if not self.FullFeedback:
            raise Exception("This model does not work with partial-feedback data")
        if type(self.parameters)==dict:
            self._K=int(self.parameters['Kappa'])
            self._W=self.parameters['Omega']
            self._D=self.parameters['Delta']
        elif type(self.parameters) in [list,np.ndarray]:
            self._K=int(self.parameters[0])
            self._W=self.parameters[1]
            self._D=self.parameters[2]
        else:
            raise Exception("Parameters should be entered correctly as a dictionairy or list (Kappa,Omega,Delta)")
        if self._K<1 or self._W>1 or self._W<0 or self._D>1 or self._D<0:
            return np.full((self._trials_,self._Num_of_prospects_), 9999)
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
        for s in range(self.nsim):
            K=np.random.randint(1,self._K+1)
            W=self._W
            D=np.random.uniform(0,self._D)
            if reGenerate:
                for p in self.prospects:
                    p.Generate()
            choices=np.zeros(self._trials_,dtype=np.int16())
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            means=np.full((data.shape[0]-1,data.shape[1]),-np.inf,dtype=np.float16)
            choices[0]=np.random.randint(0,self._Num_of_prospects_)
            grand_choices[s,0,choices[0]]=1
            ws=np.random.choice(a=[0,0.5],p=[1-W,W],size=self._trials_)
            for i in range(self._trials_-1):
                consider=set({np.random.randint(0,self._Num_of_prospects_)})
                remaining=np.array(list(set(range(self._Num_of_prospects_))-consider))
                consider|=set(remaining[np.where(np.random.uniform(size=self._Num_of_prospects_-1)>D**((i+1)/(i+2)))[0]])
                consider=np.array(list(consider))
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
    def Predict(self, reGenerate=True):
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
        self._A=self.parameters['Alpha']
        if choice_rule.lower() not in ['best','e-greedy','egreedy','e_greedy']:
            raise Exception("Please provide a choice rule (best or e-greedy)")
        self._choice_rule_=choice_rule
    def Predict(self, reGenerate=True):
        if not self.FullFeedback:
            raise Exception("This model does not currently work with partial-feedback data")
        if self._choice_rule_.lower()=='best':
            if type(self.parameters)==dict:
                self._A=self.parameters['Alpha']
            elif type(self.parameters) in [list, np.ndarray]:
                self._A=self.parameters[0]
            else:
                self._A=self.parameters
            if self._A<0 or  self._A>1: #If parameters are out of bound
                self._pred_choices_=np.full_like(self._pred_choices_, 9999)
                return self._pred_choices_
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                if reGenerate:
                    for p in self.prospects:
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
        elif self._choice_rule_.lower() in ["egreedy","e-greedy","epsilon-greedy"]:
            if type(self.parameters)==dict:
                self._A=self.parameters['Alpha']
                self._E=self.parameters['Epsilon']
            elif type(self.parameters) in [list, np.ndarray]:
                self._A=self.parameters[0]
                self._E=self.parameters[1]
            else:
                raise Exception("Parameters must be inside a dictionary or a list")
            if self._A<0 or  self._A>1 or self._E<0 or self._E>1:  #If parameters are out of bound
                self._pred_choices_=np.full_like(self._pred_choices_, 9999)
                return self._pred_choices_
            grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_),dtype=np.int16)
            for s in range(self.nsim):
                if reGenerate:
                    for p in self.prospects:
                        p.Generate()
                data=np.vstack([x.outcomes for x in self.prospects]).transpose()
                Qs=np.zeros_like(data)
                choices=np.zeros(self._trials_,dtype=np.int16)
                for i in range(self._trials_):
                    if i==0 or np.random.rand()<self._E:
                        choices[i]=np.random.randint(0,self._Num_of_prospects_)
                    else:
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
    def Predict(self, reGenerate=True):
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
                if reGenerate:
                    for p in self.prospects:
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