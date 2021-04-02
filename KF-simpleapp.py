#Implementation of Kalmann filter for plane localization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def KalmannGain(Pp,R):
    step1=Pp*np.identity(2)
    step2=np.identity(2)*Pp*np.identity(2).T+R
    K = np.zeros(step1.shape)
    #WARNING : I need to do that because my problem is simplified so i have a diagonal matrix
    # so 0/0 is impossible, i so have to divide only diagonal value.
    for i in range(len(step1)):
        a=step1[i][i]
        b=step2[i,i]
        K[i][i]=a/b
    return K

def calcPredictedState(x,A,B):
    step1= A*x
    step2= B*Ax
    return step1+step2

def calcPredictedProcess(P,A):
    predictedP=A*P*A.T+0#Qk=0
    simPredictedP=np.diag(np.diag(predictedP))
    return simPredictedP

def calcNewObsveration(measurement,Zk = 0):
    result = np.identity(2)*measurement.T + Zk
    return result


def plotKF(x,xp,measurement):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in range(len(measurement)-1):
        ax1.scatter(i,x[i][0,0], s=10, c='b', marker="s")
        ax1.scatter(i+1,xp[i][0,0], s=10, c='g', marker="s")
        ax1.scatter(i, measurement[i][0], s=10, c='r', marker="o")
        ax2.scatter(i,x[i][1,0], s=10, c='b', marker="s")
        ax2.scatter(i+1,xp[i][1,0], s=10, c='g', marker="s")
        ax2.scatter(i, measurement[i][1], s=10, c='r', marker="o")
    ax1.scatter(i+1,x[i+1][0,0], s=10, c='b', marker="s")
    ax1.scatter(i+1, measurement[i+1][0], s=10, c='r', marker="o")
    ax2.scatter(i+1,x[i+1][1,0], s=10, c='b', marker="s")
    ax2.scatter(i+1, measurement[i+1][1], s=10, c='r', marker="o")
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=4)]
    plt.legend(custom_lines, ['KF', 'Xpredicted', 'Observation'],loc='upper left')
    plt.show()

x=[]#state 
xp=[]#predicted state
P=[] #state cov
Pp=[]#predicted state cov mat
Y=[]#observations with possible error of communication or sensor issue etc...
measurement = [[4000,280],\
    [4260,282],\
    [4550,285],\
    [4860,286],\
    [5110,290]]
dt = 1 #sec
Ax = 2 #m/s²
#observation errors 
errX=25#m
errVx=6#m/s
R=np.matrix([[errX**2,0],[0,errVx**2]])
#process errors
errPx=20#m
errPVx=5#m/s
#matrix A,B,...
A = np.matrix([[1,dt],[0,1]])
B = np.array([[0.5*dt**2],[dt]])
#initial state
x.append(np.matrix(measurement[0]).T)
Y.append(np.matrix(measurement[0]).T)
P.append(np.matrix([[errPx**2,0],[0,errPVx**2]]))
i=0
for i in range(len(measurement)-1):
    #calculate predicte]d state
    xp.append(calcPredictedState(x[i],A,B))#wk=0
    #calculate predicted process cov matrix
    Pp.append(calcPredictedProcess(P[i],A))#Qk=0
    #calculate Kalmann Gain
    K = KalmannGain(Pp[i],R)
    #calc new observation
    Y.append(calcNewObsveration(np.matrix(measurement[i+1])))
    #calc current state
    x.append(xp[i]+K*(Y[i+1]-np.identity(2)*xp[i]))
    #updating process covariance mat
    P.append((np.identity(2)-K*np.identity(2))*Pp[i])
    print(x[i+1][0,0])
    print(xp[i][0,0])
    print(measurement[i+1])
    print(i)
#Plotting the scatter plots
plotKF(x,xp,measurement)
