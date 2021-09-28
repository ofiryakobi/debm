import numpy as np
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
        if callable(self._PopFun):
            return(np.mean([self._PopFun(*self._PopPars[0],**self._PopPars[1]) for i in range(EVsims)]))
        else:
            return(np.mean(self.outcomes))
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
