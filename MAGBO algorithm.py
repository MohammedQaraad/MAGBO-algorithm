#!/usr/bin/env python
# coding: utf-8

# In[4]:


# A modified gradient search rule based on the quasi-newton method and a new local search technique to improve the gra-dient-based algorithm: solar photovoltaic parameter extraction
# More details about the algorithm are in [please cite the original paper ]
# Bushra Shakir Mahmood, Nazar K. Hussein , Mansourah Aljohani ,Mohammed Qaraad.
# Mathematics,  2023


import random
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt


# Modifided Gradient Search Rule
def GradientSearchRule(ro1,Best_X,Worst_X,X,Xr1,DM,eps,Xm,dim,Flag):
    #dim = numpy.size(X,2)
    Delta = 2*random.random()*abs(Xm-X);                          #  % Eq.(16.2)
    Step = ((Best_X-Xr1)+Delta)/2;                     #    % Eq.(16.1)
    DelX = numpy.random.random((dim))*(abs(Step));                     #   % Eq.(16)
    GSR = (Worst_X-Best_X)*(DelX)/(2*(Best_X+Worst_X-2*X));   #% Gradient search rule  Eq.(15)
    if Flag == 1:
      Xs = X - GSR + DM      #                                  % Eq.(21)
    else:
      Xs = Best_X -GSR + DM 
    yp = (0.5*(Xs+X)+random.random()*DelX); #                   % Eq.(22.6)
    yq = (0.5*(Xs+X)-random.random()*DelX); #                   % Eq.(22.7)
    GSR =(yq-yp)*(DelX)/(2*(yp+yq-2*X));        #    % Eq.(23)    
    return GSR
def _mutation__(current_pos, new_pos,dim,crossover_ratio,lb,ub):
    pos_new = numpy.where(numpy.random.uniform(0, 1, dim) < crossover_ratio, current_pos, new_pos)
    return pos_new          

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step          
          
          

def objective_Fun (x):
    return 20+x[0]**2-10.*np.cos(2*3.14159*x[0])+x[1]**2-10*np.cos(2*3.14159*x[1])

def MAGBO(objf,lb,ub,dim,N,Max_iter):

   
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
  
    Cost=numpy.full(N,float("inf")) #record the fitness of all slime mold
    # initialize the location and Energy of the rabbit
    
    X = numpy.zeros((N, dim))
    for i in range(dim):
        X[:, i] = numpy.random.uniform(0,1, N) * (ub[i] - lb[i]) + lb[i]
    
    for i in range(0,N):
        Cost[i]=objf(X[i,:])
    SmellOrder = numpy.sort(Cost)  #Eq.(2.6)
    SmellIndex=numpy.argsort(Cost)
    Worst_Cost = SmellOrder[N-1];
    Best_Cost = SmellOrder[0];
    sorted_population=X[SmellIndex,:]
    Best_X=sorted_population[0,:]
    #print(" Best_X " , Best_X)
    Worst_X=sorted_population[N-1,:]
    #print(" Worst_X " , Worst_X)
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)
    
    ############################


   
    ############################
    it=0  # Loop counter
    
    # Main loop
    while it<Max_iter:
        beta = 0.2+(1.2-0.2)*(1-(it/Max_iter)**3)**2    #                       % Eq.(14.2)
        alpha = abs(beta*math.sin((3*math.pi/2+math.sin(3*math.pi/2*beta))));  #            % Eq.(14.1)
          
        for i in range(0,N):
            rand = random.sample(range(N - 1), 5)
            r1 = int(rand[0])   
            r2 = int(rand[1])       
            r3 = int(rand[2])       
            r4 = int(rand[3])       
              
            Xm = (X[r1,:]+X[r2,:]+X[r3,:]+X[r4,:])/4;                 #  % Average of Four positions randomly selected from population        
            ro = alpha*(2*random.random()-1);
            ro1 = alpha*(2*random.random()-1);        
            eps = 0.005-3*random.random();                  #               % Randomization Epsilon
        
            DM = random.random()*ro*(Best_X-X[r1,:]);
            Flag = 1;           #        % Direction of Movement Eq.(18)
            GSR=GradientSearchRule(ro1,Best_X,Worst_X,X[i,:],X[r1,:],DM,eps,Xm,dim,Flag)  
            #print("GSR " , GSR)
            DM = random.random()*ro*(Best_X-X[r1,:]);
            X1 = X[i,:] - GSR + DM;                                #     % Eq.(25)
        
            DM = random.random()*ro*(X[r1,:]-X[r2,:])
            Flag = 2
            GSR=GradientSearchRule(ro1,Best_X,Worst_X,X[i,:],X[r1,:],DM,eps,Xm,dim,Flag) 
            DM = random.random()*ro*(X[r1,:]-X[r2,:]);
            X2 = Best_X - GSR + DM;         #                            % Eq.(26)            
            Xsalp=numpy.zeros(dim)
            Xnew=numpy.zeros(dim)
            
            ro=alpha*(2*random.random()-1);   
            X3=X[i,:]-ro*(X2  -  X1); 
            ra=random.random();
            rb=random.random();
            Xnew=numpy.zeros(dim)
            Xnew = ra*(rb * X1 +(1-rb)* X2 )+(1-ra) *X3;
#             for j in range(0,dim):
#                 ro=alpha*(2*random.random()-1);                       
#                 X3=X[i,j]-ro*(X2[j]-X1[j]);           
#                 ra=random.random();
#                 rb=random.random();
#                 Xnew[j] = ra*(rb*X1[j]+(1-rb)*X2[j])+(1-ra)*X3;  #   % Eq.(27) 
    
     ## Local escaping operator(LEO)                              % Eq.(28)
            Xp=numpy.zeros(dim)
            LC=0.7
            LC = 4*LC*(1-LC);
            if random.random()<0.5 :
#                 k = math.floor(PopSize*random.random())
                Z=Levy(dim)
                ids_except_current = [_ for _ in range(N) if _ != i]
                id_1, id_2 , k = random.sample(ids_except_current, 3)
#                 L2=rand<0.5;
                
#                 Xp = Z * Xnew + random.random() * X[k,:]; 
                
                Xp  = Xnew  + Z  * (random.random() * (X[id_1, :] - X[id_2, :]))/2
                
                if random.random()< 0.5:
                    Xnew = Xnew + ((Best_X-Xp)+(random.random())*(X[id_1,:]-X[id_2,:]));    
                else:
                    Xnew = Best_X +((Best_X-Xp)+(random.random())*(X[id_1,:]-X[id_2,:])); 

            Xnew = _mutation__(X[i, :], Xnew,dim,0.1,lb,ub)
            Xnew=numpy.clip(Xnew, lb, ub)
            Xnew_Cost=objf(Xnew)
            if Xnew_Cost<Cost[i]:
                Cost[i]=Xnew_Cost 
                X[i,:]=Xnew
                if Cost[i]<Best_Cost:
                    Best_X=X[i,:]
                    Best_Cost=Cost[i]
            if Cost[i]>Worst_Cost:
                Worst_X=X[i,:]
                Worst_Cost=Cost[i]
        convergence_curve[it]=Best_Cost
       
          #if (it%1==0):
                 #print(['At iteration '+ str(it+1)+ ' the best fitness is '+ str(Best_Cost)]);
        it=it+1
                 
 
    

    return convergence_curve


Max_iterations=50  # Maximum Number of Iterations
swarm_size = 30 # Number of salps
LB=-10  #lower bound of solution
UB=10   #upper bound of solution
Dim=2 #problem dimensions
NoRuns=100  # Number of runs
ConvergenceCurve=np.zeros((Max_iterations,NoRuns))
for r in range(NoRuns):
    result = MAGBO(objective_Fun, LB, UB, Dim, swarm_size, Max_iterations)
    ConvergenceCurve[:,r]=result
# Plot the convergence curves of all runs
idx=range(Max_iterations)
fig= plt.figure()

#3-plot
ax=fig.add_subplot(111)
for i in range(NoRuns):
    ax.plot(idx,ConvergenceCurve[:,i])
plt.title('Convergence Curve of the MAGBO Optimizer', fontsize=12)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

