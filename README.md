# A Python package for Decision from Experience Behavior Modeling (DEBM)

DEBM is a **playground** for modeling behavior in the decision from experience paradigm (Barron & Erev, 2003), developed by Dr. Ofir Yakobi. 

The rational behind DEBM is to  
----
(1) Make behavioral modeling science more replicable.  
(2) Make modeling accessible for people with limited programming skills.  
(3) Be a central repository for published models.  
(4) Speed up the process of comparing, assessing and developing behavioral models.  
(5) Facilitate teaching and intuition of decision-making models.  

DEBM contains the basic building blocks of behavioral models, alongside built-in models from the decision-making literature.  
The full code is available above in the present GitHub repository.  

Contribute
====
Add your own model
----
Follow this tutorial to develop your own model, or re-write an existing model into the DEBM package.
Currently, only published models, or those in the process of publishing (e.g., submitted papers), will be included in the package.
Send your code to ofiryakobi+debm (at) gmail.com with a reference to a pre-print or the published paper describing the model.

Correct existing models, suggest improvements
----
Did you find an error in one of the models, a result that does not replicate, or found a more efficient way to write an existing code?
Please email me at ofiryakobi+debm (at) gmail.com

Make sure you are always up to date
----
**Important** - one of the essential goals for this package is to keep coding free of errors and bugs.  
If code is being corrected, you will not be able to enjoy it until you update your existing package.
To do that, make sure once in a while to update the package by going to the command line, and running  
`pip install debm -U`


Share and ask for help
----
Be part of the Google group community: suggest, ask, consult, and help others:  
https://groups.google.com/g/debm_package

# Hands-on tutorial for non Python programmers

The basics
====
Using an existing model (whether it was created by you, or one of the built-in models) is easy.

As an example, we will use the sample-of-k model, also known as Naive Sampler (Erev & Roth, 2014).
This model asserts that when making repeated decisions, the agents recall a small sample of past experiences of the size *k* to evaluate the different prospects.
The sampling process is done with replacement from all previous experience.

As an example, we will replicate the predictions of Erev and Roth (2014), problems 1 and 2 (Table 2).

Installing debm
----
If you don't have Python installed, I recommend doing so by [installing Anaconda](https://www.anaconda.com/products/individual).  
Go to command line (e.g., Start -> run -> cmd. If you use Anaconda: Start -> Anaconda prompt).  
Type `pip install debm`

Setting up
----
Using your favorite Python IDE (I recommend installing Anaconda and using Spyder), we will now import *debm* and define the environment: the prospects and the model.
The hash tag symbol is used for commenting, and will be used here to clarify the code.

```# We will use Prospect and Sample_of_K for now, the rest will be used later  
from debm import Prospect, Sample_of_K, resPlot, FitMultiGame, makeGrid, saveEstimation, resPlot
import numpy as np #We will use NumPy as well, with its short alias np
```

We will start by defining the two prospects simulating 100 trials decision-making problem described in Erev and Roth (2014).  
This problem is a repeated choice between two prospects: a status quo which gives 0 (zero) all the time, and a risky prospect  
that gives 1 most of the time (90%) or -10.  

We will define the status quo, and name it accordingly:
`StatQuo=Prospect(100,[0]*100,False)`
This line of code creates a Prospect named StatusQuo, with 100 trials (the 1st argument).  
The 2nd argument is an array of 100 zeros (instead of writing [0,0,0,0,0....,0]) - this is the outcomes array,  
stating that the outcome is zero in every trial.    
The 3rd argument passed to Prospect is boolean (True/False), and it is used when we pass a function (see below).  
  
Now we will define a prospect that produce -10 (minus 10) 10% of the time, and +1 (a gain of one point) 90% of the time.  
For that, we will use NumPy's (a popular Python package) function - np.random.choice.  
The syntax is np.random.choice(a, size, replace, p):  
a – the array of possible outcomes; size – how many samples to draw;  
replace – draw with replacement? (True/False). If false – the size should be equal or less than the size of the array a;  
p – an array of probabilities in the size of a. Let’s consider the example below:  
np.random.choice([1,-10],1,True,[0.9,0.1])
The result would be one number, either 1 (in probability 0.9) or -10.  
If we want to output 100 numbers instead of one: np.random.choice([1,-10],1,True,[0.9,0.1]).  
  
The formal syntax for a Prospect is `Prospect(trials, fun, oneOutcomePerRun, *args, **kwargs)`  
trials - the number of trials the decision making problem.  
fun - an array of outcomes (in the size of *trials*), or a function.  
oneOutcomePerRun - boolean (True/False), stating whether the function above returns  
one value per run, or returns *trials* outcomes (e.g., 100 outcomes) at once.  
*args,**kwargs - optional arguments to be passed to the function.  

When creating our *B1* prospect, we will pass a function (and corresponding arguments), as follows:
`B1=Prospect(100,np.random.choice,False,[-10,1],100,True,[0.1,0.9])`  

We defined a prospect named B1 with 100 trials, passed the function np.random.choice,  
Passed False to state that the function we are passing generates all trials at once (not one by one),  
Then we passed the arguments for `np.random.choice` as before: [-10,1],100,True,[0.1,0.9].  
  
Now we will define the Sample_of_K model, we will name it sok1 (sample of k, problem 1) for convenience.  
The syntax for Sample_of_K is Sample_of_K(parameters, prospect_list, number_of_simulations).  
Let's consider the code below:
`sok1=Sample_of_K({'Kappa':5},[StatQuo,B1],1000)`
Parameters is a dictionary of parameters. In this specific model we only have Kappa, which we defined as 5 (as in Erev & Roth 2014).  
[StatQuo,B1] - a list of the two prospects we defined.  
1000 - the number of simulations to run each time we generate predictions.  

That's it! We set up a model, and your code should look like the following at this point:  
```
from debm import Prospect, Sample_of_K, resPlot, FitMultiGame, makeGrid, saveEstimation ,resPlot
import numpy as np #We will use NumPy as well, with its short alias np

StatQuo=Prospect(100,[0]*100,False)
B1=Prospect(100,np.random.choice,False,[-10,1],100,True,[0.1,0.9])
sok1=Sample_of_K({'Kappa':5},[StatQuo,B1],1000)
```

Now let's make predictions, and save them to a new variable called *choices1*:  
`choices1=sok1.Predict()`  
*choices1* stores the choice rates predicted for each prospect for the 100 trials we defined, based on 1000 simulations.  
You can `print(choices1)` to inspect the results.  
To see the mean over all trials, type `choices1.mean(axis=0)` (axis=0 states that we want the mean over rows [trials] and not the grand mean).  
The results should be approximately 38% and 62% as in Erev and Roth.  

We can easily plot the results:  
`sok1.plot_predicted()`
If you want to aggregate over blocks, you can simply type the number of blocks:  
`sok1.plot_predicted(4)` (just make sure the number of trials is dividable by the number of blocks)  
Regardless to the names you gave each prospect, they will be named in the figure according to their order in the model (A, B, C...Z).  
In our case, StatusQuo was entered first and corresponds to *A*.  


Fitting
----

If we want to estimate the parameters of the model, we need real human data observations.  
Download the following example [https://github.com/ofiryakobi/debm/blob/master/help/ex1.csv](https://github.com/ofiryakobi/debm/blob/master/help/ex1.csv) to your computer.  
Make sure you know the path of the file you downloaded (e.g., c:\users\myName\downloads\).  

Add the following line of code to your script, and change the path to the real file path:  
`observed1=np.genfromtxt('c:\\users\\myName\\downloads\\ex1.csv', delimiter=',')`  

**Note**: Python does not like \\ (one backslash) - use two as in the code line above.  

After running this line of code, you will have a variable called *observed1*, containing the aggregated observed choice rates of 86 participants (in this case they are simulated).  
Use `print(observed1)` to inspect the choice rates.  

We will store them in our model:  
`sok1.set_obs_choices(observed1)`  

Now we can even plot them, using a similar syntax to the one we used earlier:  
`sok1.plot_observed()`
or, e.g.:  
`sok1.plot_observed(4)`

We can even plot the fit of the current predicted choices (the ones we generated earlier, with Kappa=5):
`sok1.plot_fit(4)`

You can see that the fit is not great. We can quantify it, using `sok1.loss(loss_function, scope)`.  
loss_function could be MSD (mean squared deviation; identical to MSE) or -LL (negative log-likelihood),  
and the scope should be either *prospectwise* (comparing one value for each prospect), or *bitwise* (comparing each trial of each prospect).  
Calculating the MSD prospectwise:
`sok1.loss('MSD','pw')`  
The current MSD is ~0.0145.

Can we find a better fit?  
Using the built-in OptimizeBF (BF stands for brute-force, trying all possible values as in a grid search).  
`OptimizeBF(pars_dicts, kwargs)`  
We need to pass a list of parameters, each packed in a dictionary as before, as well as arguments to the loss function.  

We can easily create a grid using the helper function *makeGrid*:  
`pspace=makeGrid({'Kappa':range(1,51)})`  
Now the variable pspace (pspace as in parameter-space) contains all possible Kappa values from 1 to 50 (**note** that in Python the upper bound is usually exclusive, so 1,5 means 1,2,3,4), in the right structure.  
To estimate Kappa we can run the following command:  
`res1=sok1.OptimizeBF(pspace,True,'MSD','pw')`

The first argument is pspace - the parameter space and values over the grid we evaluate.  
The second is boolean (True/False) stating if we want to see a progress bar during the process.  
The next two arguments go directly to the *loss* function we used before - telling it the loss function and scope we would like to use for fitting.  
We save the results in a new variable we called *res1*.  

res1 is a dictionary, containing the best set of parameters found, the corresponding minimum loss value, and a list of loss values for each iteration.  
The following command will pring the best fitted Kappa:  
`print(res1['bestp'])`  # We found Kappa=3 was the best representation of the data

We can use the helper function *resPlot* to plot the loss function over iterations (useful mostly for 1 or 2 parameter models):  
`resPlot(res1)`

It is possible to plot the fit (visualizing the observed and predicted choice rates), but first we will need to update our predictions:  
```
sok1.parameters={'Kappa':3} # Set Kappa according to the estimation
sok1.Predict() # Generate new predictions with Kappa=3
```

Now we can plot the fit, e.g. over five blocks of 20 trials:
`sok1.plot_fit(5)`

To save the estimation's results to a csv file, use:  
`saveEstimation(res1,'c:\\your_desired_path\\results.csv')`  


More advanced user can use external packages (e.g. SciPy) for optimization, using the CalcLoss function which accepts parameters and returns the loss.
e.g. `res=minimize(modelname.CalcLoss,[0],('MSE','pw'))`  


Individual differences
----
The example above demonstrated fitting over the aggregated choice rates of 84 participants.  
We can also fit Kappa for each individual, using the same mechanism. However, an extra layer of Python programming is required.  
(note: I plan to add a built-in functionality for individual differences estimation in future release)  

Download the data of three subjects:  
[Subject 1](https://github.com/ofiryakobi/debm/blob/master/help/sub1.csv)  
[Subject 2](https://github.com/ofiryakobi/debm/blob/master/help/sub2.csv)  
[Subject 3](https://github.com/ofiryakobi/debm/blob/master/help/sub3.csv)  

To make the estimation faster (for demonstration only), we will limit the values of Kappa to 1-10.  

```
path='C:\\your\\path\\' # Set the path for the downloaded files, remember double backslashes
pspace=makeGrid({'Kappa':range(1,11)})
individual_kappas=[] # We will store the Kappas we found in this array
for file in ['sub1.csv','sub2.csv','sub3.csv']: # A loop going through the different subject data files
    observed=np.genfromtxt(path+file,delimiter=',') # This NumPy function reads the relevant csv file in each loop iteration, and stores it in the variable observed
    sok1.set_obs_choices(observed) # Store the observed choices in the model
    tmp_res=sok1.OptimizeBF(pspace,True,'MSD','pw')  # Estimate 
    individual_kappas.append(tmp_res['bestp']['Kappa']) # Append the best Kappa from the results variable tmp_res into individual_kappas array
```

individual_kappas is an array which now contains the best Kappas found for subject 1,2,3. The values should be 8, 3, 6.

More functionality
====
Another example, demonstrating parameter recoverability, inspecting prospects, multi-game fitting and external functions for estimation
----
Here we will use a Reinforcement learning model with a Q-delta learning rule (Sutton & Barto, 1998).  
This model implies that people update their expectancies for each prospect based on previous feedback, and their learning rate is   
determined by Alpha (ranges 0 to 1), where low values represent slow learning rates (small weight for new information) and high values fast learning rates.

This model is also included with the basic models supplied with *debm*, we can import it by adding RL_delta_rule to the import statement we already have in our code,  
or add another line (for orgnization reasons, it is better to place import statements at the beginning of the script):  
`from debm import RL_delta_rule`

Let's start by defining our prospects:  
```
trials=100
A=Prospect(trials,np.random.choice,False,[3,0],trials,True,[0.45,0.55])
B=Prospect(trials,np.random.choice,False,[1.6,-20],trials,True,[0.97,0.03])
C=Prospect(trials,np.random.choice,False,[1.7,-20],trials,True,[0.94,0.06])
```

Note that instead of writing "100" every time, we stored it in a variable called *trials* and then passed to the Prospects this variable.  
If we want to change the number of trials, we can just change the value in this variable, and not each instance of a related model or prospect.  
Our prospects could be formulated as A: 3, p=0.45, 0 otherwise; B: 1.6, p=0.97, -20 otherwise; and C: 1.7, p=0.94, -20 otherwise.  

We can inspect these prospects in two ways. First we can use `prospect.EV()` to get the expected value (based on 10,000 simulations).  
Next, we can compare the two prospects in terms of the proportion each prospect yields better results.  
```
print(A.EV(), B.EV(), C.EV())  # You can skip the print command if you run one line of code at the time, just A.EV()  
C>B  
```
You will note that prospect C is better than B most of the time, but its expected value is worse.  

Next, we define the mode. This model (RL_delta_rule) has another argument we need to pass, called "choice rule".  
Currently, there is only 'best' choice rule available (i.e., agents choose the best of the options based on the Q value in each step).  
In the future, other choice rules such as *epsilon-greedy* will be added.  

```
myModel=RL_delta_rule('best',{'Alpha':0.5}, [A,B,C], 1000)
```

After 'best', we pass the model parameters (in this case - Alpha), the prospect, and number of simulations.  

Side note that we don't have to pass the value of alpha if we only want to estimate (the estimation algorithm will set new values each time).  
In this case you can use (the name of the parameter must be defined, and some value - even None, should be set):
```
myModel=RL_delta_rule('best',{'Alpha':None}, [A,B,C], 1000)
```

If we run `myModel.Predict()` we will get a nasty error, because Alpha is None - so no predictions can be made.  

First, we will define two models and simulate predictions of agents with Alpha of 0.1, and 0.8.
```
myModel1=RL_delta_rule('best',{'Alpha':0.1}, [A,B,C], 1000)
myModel2=RL_delta_rule('best',{'Alpha':0.8}, [A,B,C], 1000)
```

We will store the predictions in two variables:  
```
RLpredictions1=myModel1.Predict()
RLpredictions2=myModel2.Predict()
```

You can inspect the two predictions using `myModel1.plot_predicted(5)` and `myModel1.plot_predicted(5)`.  
Note the difference in the shapes of the learning curves.  

Next, we will define a new model and try to estimate each set of predictions to see if we can recover the two parameters correctly.  
```

estimateRL=RL_delta_rule('best',{'Alpha':None}, [A,B,C], 1000)
estimateRL.set_obs_choices(RLpredictions1)
pspace=makeGrid({'Alpha':np.arange(0,1.01,0.01)})
RL_est_results1=estimateRL.OptimizeBF(pspace,True,'MSE','pw')

estimateRL.set_obs_choices(RLpredictions2)
RL_est_results2=estimateRL.OptimizeBF(pspace,True,'MSE','pw')

```

At this point you should be able to extract the best parameters set from each of the results dictionaries.  
You can also visually inspect them using `resPlot(RL_est_results1)` and `resPlot(RL_est_results2)`.  

Success, we recovered the correct Alphas (0.1 and 0.8).  


For more advances users, if you would like to use external optimization algorithms, you can do that with CalcLoss method.  
see below an example:  
```
res=Scipy.Optimize.minimize(estimateRL.CalcLoss,[0],('MSE','pw'),method="COBYLA")
```


In some cases, participants complete more than one decision-making task, and we want to find the best parameters that fit the data from these tasks.  
Recall our prospects from above:  
```
A=Prospect(trials,np.random.choice,False,[3,0],trials,True,[0.45,0.55])
B=Prospect(trials,np.random.choice,False,[1.6,-20],trials,True,[0.97,0.03])
C=Prospect(trials,np.random.choice,False,[1.7,-20],trials,True,[0.94,0.06])
```
Let's say that instead of playing one tasks (choosing between A,B and C) participants play two games:  
A, B and B, C.  

We will define these models accordingly:  
```
game1=RL_delta_rule('best',{'Alpha':None}, [A,B], 1000)
game2=RL_delta_rule('best',{'Alpha':None}, [B,C], 1000)
```

We will generate agents that play these two games with an alpha of 0.65, and try to recover alpha.  
```
generate_agents1=RL_delta_rule('best',{'Alpha':0.65}, [A,B], 1000)
generate_agents2=RL_delta_rule('best',{'Alpha':0.65}, [B,C], 1000)
agents1=generate_agents1.Predict()
agents2=generate_agents2.Predict()
```

Now we create the estimation models:  
```
estimateRL1=RL_delta_rule('best',{'Alpha':None}, [A,B], 1000)
estimateRL2=RL_delta_rule('best',{'Alpha':None}, [B,C], 1000)
```

Now for the new object we need to get familiar with: FitMultiGame.  
It's pretty straight-forward, we pass the array of models (or games), and the array of "observations".  
```
fitted_over_two_Tasks=FitMultiGame([estimateRL1,estimateRL2],[agents1,agents2])
```

From here, fitting is similar to single task fitting:  
```
res_fit_2tasks=fitted_over_two_Tasks.OptimizeBF(pspace, True, 'MSE', 'PW')
```
Note that we used pspace (the parameters grid we defined earlier), but you may want to re-define it to test other values.  

You should see in *res_fit_2tasks* that the best Alpha is indeed 0.65.  

Importing data from a csv file using Pandas
----
If you are not familiar with Python/Pandas, this section could be helpful.  
Pandas, like NumPy, is a very popular and useful package.  
It allows working efficiently with dataframes.  
[More resources on Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/tutorials.html).  
To import Pandas:  
```
import pandas as pd
```

You can import csv or excel files from your local machine, or from a URL. Let's import data from Yakobi et al. (2020), experiment 1.  
In this experiment, each one of the 85 participants completed 3 decision tasks.  

```
#Read the csv file directly from the web
df=pd.read_csv('https://github.com/ofiryakobi/debm/raw/main/help/TAXING_RECKLESS_BEHAVIORS_data.csv')

#In he column named "player.condition", replace the values noR, overR and modR,
#with 0, 0.4 and 0.8, respectively. This is just for convenience.
df['player.condition'].replace({'noR':0,'overR':0.8,'modR':0.4},inplace=True)

#The experiment included 3 games with 100 trials each, but the numbering of trials
#is 1-300. Make the trials 1-100 for each task, save the results in a column named
#trial.
df['trial']=df['subsession.round_number']-100*((df['subsession.round_number']-1)//100)

#Aggregate the dataframe over condition and trial and save it in a new dataframe
#named df_agg. Take the mean of the relevant columns only:safe_choice, 
#medRisk_choice, and highRisk_choice.
df_agg=df.groupby(['player.condition','trial'])[['safe_choice','medRisk_choice','highRisk_choice']].mean()

#Convert and store each of the three tasks (0,0.4,0.8) into an array.
obs_0=df_agg.loc[0.0].to_numpy()
obs_04=df_agg.loc[0.4].to_numpy()
obs_08=df_agg.loc[0.8].to_numpy()
```  

That's it! We now have three arrays, each containing the mean choice rates  
for 85 participants in each of the 100 trials. We can now use them as observations  
to fit models.  

Below (section "Creating dependent (or correlated) prospects"), you will learn how to  
create correlated prospects (which is needed to the Yakobi et al. study).


Creating dependent (or correlated) prospects
----

In some cases, we will want to model environments in which the prospects are correlated.  
Let's look at an example of three prospects:  
A- 10, p=0.9; -100 otherwise  
B- 8, p=0.8; -80 otherwise  
C- 8 if the outcome of B is 8; otherwise, draw a random number  
from a uniform distribution between -180 and 20.  

As you noted, the outcome of C in each trial is dependent on the trial outcome of B.  
In order to implement this design, we can take advantage of the fact that *Prospect*  
objects accept functions as inputs.  
We will write a Python function that accepts a Prospect, inspect its outcomes, and returns  
another prospect accordingly.  
Read the comments (hashtags) inside the code to better understand its mechanism.  

```
def dependentOutcomes(prospect): #def states that we define a function, called dependentOutcomes  
#This function accepts an input which we named prospect  
    prospect.Generate() # a Prospect object has a method called generate.  
	# When we run that (e.g., A.Generate()), we ask the prospect to draw  
	# a new outcomes matrix.  
    output=prospect.outcomes.copy() #Copy the outcomes to a new variable called output
    output[output!=8]=np.random.randint(-180, 21) #Wherever the value in output is NOT 8,  
	# store a random draw (we used NumPy again) between -180 and 21 (exclusive).  
    return output  # Return the output matrix
```
In summary, this function accepts a prospect, generates new values for this prospect,  
copy these values to a new variable - and replace all values that are not *8* with a  
number drawn from U[-180,20], using NumPy's np.random.randint function.  

You should be able by now to define prospects A and B by yourself:  
```
trials=100 # In case trials is not already defined somewhere else in your code
A=Prospect(trials,np.random.choice,False,[10,-100],trials,True,[0.9,0.1])
B=Prospect(trials,np.random.choice,False,[8,-80],trials,True,[0.8,0.2])
```

Now for the tricky part, prospect C. Instead of using a NumPy function to generate  
random values, we will use the *dependentOutcomes* function we wrote earlier, and pass  
prospect *B* as its arguments.  
```
C=Prospect(trials,dependentOutcomes,False,B)
```

The full code for this section:  
```
def dependentOutcomes(prospect):
#This function accepts an input which we named prospect  
    prospect.Generate() 
    output=prospect.outcomes.copy()
    output[output!=8]=np.random.randint(-180, 21)
    return output  
trials=100 # In case trials is not already defined somewhere else in your code
A=Prospect(trials,np.random.choice,False,[10,-100],trials,True,[0.9,0.1])
B=Prospect(trials,np.random.choice,False,[8,-80],trials,True,[0.8,0.2])
C=Prospect(trials,dependentOutcomes,False,B)

```


Creating dynamic prospects
----

Until now we created prospects that do not change over time (trials).  
If we want to create a dynamic prospect, we need to create a function that generates outcomes.  
Consider the following prospect:  
D- 10, p=0.9-x; -100 otherwise, where x=trial/200.  

In other words, the probability depends on the trial such that the task becomes riskier with time  
(the probability to lose 100 points increased).  
Let's program it:  

```
def riskyWithTime(trials): # This function accepts trials - the number of trials
    outcomes=np.zeros(trials) # Create outcomes - an array in the size of trials
    for t in range(trials): # Loop over the following code trials times
        #using t as the index (e.g., it will start with 0 and end in 99 if there are 100 trials)
        #The line below generates 10 or -100 in probabilities the depend on the current trial
        #using NumPy random.choice, and stores it in the corresponding place in outcomes array 
        outcomes[t]=np.random.choice([10,-100],p=[0.9-t/200, 0.1+t/200])
    return outcomes # That's it - return the results
    
D=Prospect(trials,riskyWithTime, False, trials)
```

After defining riskyWithTime, we create a prospect D, now using our own  
function and not NumPy. We state False because our function returns a matrix with all trials at once,  
and not just one trial at a time. Then we pass *trials* to our function (which expects this input).  
You can use D.plot() to see how the outcomes unfold over trials or blocks.  


Using the reGenerate option
----
As you have seen earlier, Models.Predict() allows generating predictions.  
In each simulation, the set of prospects is forced to generate new outcomes,  
which makes sense if our outcomes are stochastic.  If we define 1000 simulations,  
we generate 1000 sets of outcomes from each prospect, and run the prediction algorithm on each.  

In some cases, we would like to simulate predictions from agents that receive the same set  
of outcomes. In that case, we want the outcomes to stay the same for the whole set of simulations.  
For that, we can specify Models.Predict(reGenerate=False) (or simply Models.Predict(False)).  
If it is not specified, the default value is True (generate new outcomes in each iteration).  



Creating your very own model
----
Finally and most importantly, you probably have your own idea of how humans make decisions.  
Why not formalizing your thought into a quantifiable, reproducible model?  

A model is an instance of a Model object. It means that whatever your model is,  
it inherits some basic features and methods from a parent object called *Model*.  
For example, by default your model will have an attribute named parameters, nsim,  
name, and more. It will have built-in functions such as OptimizeBF (that we have seen earlier).  
It is convenient because you don't have to program all of it from scratch, and also all models  
in *debm* share the same functions and attributes.  

Let's create a simple one-parameter model that assumes the following:  
(1) People only care about the **last** set of outcomes they experienced.  
(2) They will choose the best outcome in probability *Gamma* (that's the free parameter, ranging 0 to 1).  
I named this model *GreedyMyopia*

We will take it line by line:  
`from debm import Model`
Import the prototype *Model* object from the *debm* package.  

`class GreedyMyopia(Model):`  
We define objects in python using the class statement. Here we state that GreedyMyopia  
is a type of *Model*.  

There are two functions (i.e., starting with *def*) that are mandatory:  
__init__(self) - note the double underscores in each side of init.  
```
    def __init__(self,*args):
        Model.__init__(self,*args)
        self.name="Greedy Myopia"
```
This function is called when an object is initiated. It initiates the  
Model object by passing arguments (e.g., parameters and prospects) to it.  
Then we define the *name* attribute and set a name for our model.  
(if you are curios about why we need to pass *self* each time, read [here](https://www.w3schools.com/python/gloss_python_self.asp)).  

The next mandatory function is *Predict* (remember - Python is case sensitive, use capital P).  
Predict has to do a few things before making predictions:  
(1) Read the parameter/s and store it/them in an appropriate attribute (which should start with an underscore).  
To make our model more versitile, we will write some code that can read two  types of parameters:  
A dictionary (what we usually use in this tutorial), or a list (a simple array).  
(2) In every iteration (simulation), generate new outcomes ONLY if reGenerate==True.  
(3) Make a predictions matrix (this is where the magic happens), save it in self._predictions,  
and return the predictions.  


```
    def Predict(self, reGenerate=True):  # reGenerate gets a default value of True if not set
        if type(self.parameters)==dict:  # Did the user input parameters as a dictionary?
            self._g=self.parameters['Gamma'] # Store the value in self._g
        elif type(self.parameters)==list: # Not a dict; was it a list?
            self._g=self.parameters[0] # Read the first value in the list into self._g
        else: # Not a list and not a dict? Raise an error to the user (stops execution)
            raise Exception("Parameters should be a list or a dictionary")        
```

The next line sets up a 3-dimensional matrix (filled with zeros, hence the name) with number of rows  
corresponding to the number of trials, and number of columns corresponding to the number  
of prospects. There are *nsim* matrices, each represents the result of one simulation.  

`grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_))`

Let's start simulating behavior, this is the main part of our model:  
```
        for s in range(self.nsim):
            if reGenerate:
                for p in self.prospects:
                    p.Generate()
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            for i in range(self._trials_):
                if i==0 or np.random.rand()>self._g:
                    choice=np.random.choice(self._Num_of_prospects_)
                    grand_choices[s,i,choice]=1
                else:
                    choice=np.argmax(data[i-1])
                    grand_choices[s,i,choice]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return self._pred_choices_
```

Let's take it line by line:  

`for s in range(self.nsim):` - loop *nsim* times  

```
            if reGenerate:
                for p in self.prospects:
                    p.Generate()
```
If the model is defined to reGenerate,  go over each prospect  
and generate new outcomes.  

`data=np.vstack([x.outcomes for x in self.prospects]).transpose()`  
This line above takes the outcomes of all prospects, and stack it one by one  
so we get a matrix (*data*) where each column is a prospect, and each row is a trial.  

```
for i in range(self._trials_):
```
Iterate over trials (remember: Python indexing starts from 0, so i==0 is the first trial).  

```
                if i==0 or np.random.rand()>self._g:
                    choice=np.random.choice(self._Num_of_prospects_)
                    grand_choices[s,i,choice]=1
```

If it is the first trial, OR a random number (between 0 and 1) is  
greater than Gamma, draw a random prospect.  
self._Num_of_prospects_ stores the number of prospects in the model, so if we have 3 prospects,
this line of code draws 0,1 or 2 and stores it in *choice*.  
Then, place *1* in the corresponding place (according to the simulation number,  
trial number, and selected prospect).

```
                else:
                    choice=np.argmax(data[i-1])
                    grand_choices[s,i,choice]=1
```
In every other case, the choice is the prospect with the highest value.  
*np.argmax* returns the **index** of the max number. For instance:  
`np.argmax([40,50,100,0,-3,10])`
Returns *2*, which is the index of 100.  
**Important**: For simplicity, we ignore cases where we have ties (more than one max).  
Whenever there are ties, the first index will be chosen.  
When you program a real model, you better address it (e.g., choose randomly between the best outcomes).  

```
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return self._pred_choices_
```
After all simulations are completed, average over simulation, store in self._pred_choices_  
and return it.  

The whole model:
```
class GreedyMyopia(Model):
    def __init__(self,*args):
        Model.__init__(self,*args)
        self.name="Greedy Myopia"
    def Predict(self, reGenerate=True):
        if type(self.parameters)==dict:
            self._g=self.parameters['Gamma']
        elif type(self.parameters)==list:
            self._g=self.parameters[0]
        else:
            raise Exception("Parameters should be a list or a dictionary")        
        grand_choices=np.zeros((self.nsim,self._trials_,self._Num_of_prospects_))
        for s in range(self.nsim):
            if reGenerate:
                for p in self.prospects:
                    p.Generate()
            data=np.vstack([x.outcomes for x in self.prospects]).transpose()
            for i in range(self._trials_):
                if i==0 or np.random.rand()>self._g:
                    choice=np.random.choice(self._Num_of_prospects_)
                    grand_choices[s,i,choice]=1
                else:
                    choice=np.argmax(data[i-1])
                    grand_choices[s,i,choice]=1
        self._pred_choices_=np.mean(grand_choices,axis=0)
        return self._pred_choices_
```

Now you can play around with the model, see what predictions you are getting  
for different Gammas, plot the results, etc. Here is a quick start for you to copy&paste:  
```
gm=GreedyMyopia({'Gamma':1}, [A,B], 1000)
predicted=gm.Predict()
print(predicted.mean(axis=0))
gm.plot_predicted()
```

Having difficulties formulating your own model?  
(1) Take a look at the code for [other models in the package](https://github.com/ofiryakobi/debm/blob/main/Models.py).  
(2) Ask for help in our [Google discussion group](https://groups.google.com/g/debm_package).  


More Tips and Tricks
----
Save the predictions of your model, for example:  
yourmodel.save_predictions("c:\\filePath\\morepath\\chooseName.csv")  

Features that will be added in the future
----
This package will be periodically updated with new features and models.  
The planned features include:  
(1) Multiprocessor support (for faster estimation).  
(2) Support of partial feedback.  
(3) Additional built-in estimation algorithms.  
(4) New models.  
(5) Built-in support for individual differences analyses and estimation.  


