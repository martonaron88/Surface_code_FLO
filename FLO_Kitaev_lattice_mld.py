import numpy as np
import matplotlib.pyplot as plt
import pymatching as pm
import pandas as pd
import scipy
import time
from matplotlib import colors
from matplotlib import cm as cmx
import sys

np.set_printoptions(threshold=np.inf)

def build_majorana_network(d):
    """
    Build up the proper Majorana network of the surface code (origianl geometry) with code distance d:
    Integer index of each Majorana is defined as: 
    starting from top left, go right and then switch row, index*4. Then add integer index of each 
    Majorana: 0, 1, 2, 3 starting from top right, going anti-clockwise.
    S stabilizers, X and Z operators therefore are defined differently in even and odd rows.
    """
    s_stabs   = np.zeros((d**2+(d-1)**2, 4), int) #stabilizers of c4 codes (qubits, -c1c2c3c4)
    z_ops     = np.zeros((d**2+(d-1)**2, 2), int) #z-operators of c4 codes (ic2c3)
    x_ops     = np.zeros((d**2+(d-1)**2, 2), int) #x-operators of c4 codes (ic1c2)
    link_ops  = np.zeros(((d**2+(d-1)**2-1)*2, 2), int) #link operators 
    logic_z = np.array([0, 4*(d-1)+3], int)
    logic_x = np.array([0, 4*(d**2+(d-1)**2-d)+1], int)
    logic_y = np.array([0, 4*(d**2+(d-1)**2-1)+2], int)
    logic_s = np.array([0, 4*(d**2+(d-1)**2-d)+1, 4*(d**2+(d-1)**2-1)+2, 4*(d-1)+3], int)
    for i in range(d-1): #fill up the operators located on qubits
        for j in range(d):
            s_stabs[i*(2*d-1)+j] = [4*(i*(2*d-1)+j),  4*(i*(2*d-1)+j)+1, 4*(i*(2*d-1)+j)+2, 4*(i*(2*d-1)+j)+3]
            x_ops[i*(2*d-1)+j]   = [4*(i*(2*d-1)+j),  4*(i*(2*d-1)+j)+1]
            z_ops[i*(2*d-1)+j]   = [4*(i*(2*d-1)+j), 4*(i*(2*d-1)+j)+3]
        for j in range(d-1):
            s_stabs[i*(2*d-1)+d+j] = [4*(i*(2*d-1)+d+j)+1,4*(i*(2*d-1)+d+j)+2, 4*(i*(2*d-1)+d+j)+3, 4*(i*(2*d-1)+d+j)]
            x_ops[i*(2*d-1)+d+j]   = [4*(i*(2*d-1)+d+j)+1, 4*(i*(2*d-1)+d+j)+2]
            z_ops[i*(2*d-1)+d+j]   = [4*(i*(2*d-1)+d+j)+1, 4*(i*(2*d-1)+d+j)]
    for j in range(d):
        s_stabs[d**2+(d-1)**2-d+j] = [4*(d**2+(d-1)**2-d+j),  4*(d**2+(d-1)**2-d+j)+1, 4*(d**2+(d-1)**2-d+j)+2, 4*(d**2+(d-1)**2-d+j)+3]
        x_ops[d**2+(d-1)**2-d+j]   = [4*(d**2+(d-1)**2-d+j),  4*(d**2+(d-1)**2-d+j)+1]
        z_ops[d**2+(d-1)**2-d+j]   = [4*(d**2+(d-1)**2-d+j), 4*(d**2+(d-1)**2-d+j)+3]
    for i in range(d-1): #fill up the link operators, its a bit tricky
        if i%2==0:
            for j in range(d-1):
                if j%2==0:
                    link_ops[4*(d-1)*i+4*j] = [4*(i*(2*d-1)+d+j), 4*(i*(2*d-1)+j)+2]
                    link_ops[4*(d-1)*i+4*j+1] = [4*(i*(2*d-1)+d+j)+1, 4*((i+1)*(2*d-1)+j)+3]
                    link_ops[4*(d-1)*i+4*j+2] = [4*((i+1)*(2*d-1)+j+1), 4*(i*(2*d-1)+d+j)+2]
                    link_ops[4*(d-1)*i+4*j+3] = [4*(i*(2*d-1)+d+j)+3, 4*(i*(2*d-1)+j+1)+1]
                else:
                    link_ops[4*(d-1)*i+4*j] = [4*(i*(2*d-1)+j)+2, 4*(i*(2*d-1)+d+j)]
                    link_ops[4*(d-1)*i+4*j+1] = [4*((i+1)*(2*d-1)+j)+3, 4*(i*(2*d-1)+d+j)+1]
                    link_ops[4*(d-1)*i+4*j+2] = [4*(i*(2*d-1)+d+j)+2, 4*((i+1)*(2*d-1)+j+1)]
                    link_ops[4*(d-1)*i+4*j+3] = [4*(i*(2*d-1)+j+1)+1, 4*(i*(2*d-1)+d+j)+3]
        else:
            for j in range(d-1):
                if j%2==0:
                    link_ops[4*(d-1)*i+4*j] = [4*(i*(2*d-1)+j)+2, 4*(i*(2*d-1)+d+j)]
                    link_ops[4*(d-1)*i+4*j+1] = [4*((i+1)*(2*d-1)+j)+3, 4*(i*(2*d-1)+d+j)+1]
                    link_ops[4*(d-1)*i+4*j+2] = [4*(i*(2*d-1)+d+j)+2, 4*((i+1)*(2*d-1)+j+1)]
                    link_ops[4*(d-1)*i+4*j+3] = [4*(i*(2*d-1)+j+1)+1, 4*(i*(2*d-1)+d+j)+3]
                else:
                    link_ops[4*(d-1)*i+4*j] = [4*(i*(2*d-1)+d+j), 4*(i*(2*d-1)+j)+2]
                    link_ops[4*(d-1)*i+4*j+1] = [4*(i*(2*d-1)+d+j)+1, 4*((i+1)*(2*d-1)+j)+3]
                    link_ops[4*(d-1)*i+4*j+2] = [4*((i+1)*(2*d-1)+j+1), 4*(i*(2*d-1)+d+j)+2]
                    link_ops[4*(d-1)*i+4*j+3] = [4*(i*(2*d-1)+d+j)+3, 4*(i*(2*d-1)+j+1)+1]
    for i in range(d-1):
        link_ops[(2*(d-1))**2+i] = [4*i+3, 4*(i+1)]
        link_ops[(2*(d-1))**2+d-1+i] = [4*i*(2*d-1)+1, 4*(i+1)*(2*d-1)]
        link_ops[(2*(d-1))**2+2*(d-1)+i] = [4*(i*(2*d-1)+d-1)+2, 4*((i+1)*(2*d-1)+d-1)+3]
        link_ops[(2*(d-1))**2+3*(d-1)+i] = [4*((d-1)*(2*d-1)+(i+1))+1, 4*((d-1)*(2*d-1)+i)+2]
    sx_ops = np.zeros((d**2+(d-1)**2, 2), dtype=int) #s*x operators, ic3c4
    sz_ops = np.zeros((d**2+(d-1)**2, 2), dtype=int) #s*z operators, ic2c3
    for i in range(d**2+(d-1)**2):
        sx_ops[i,0] = np.delete(s_stabs[i], [np.where(s_stabs[i]==x_ops[i][0]), \
				np.where(s_stabs[i]==x_ops[i][1])],  axis=0)[1]
        sz_ops[i,0] = np.delete(s_stabs[i], [np.where(s_stabs[i]==z_ops[i][0]), \
				np.where(s_stabs[i]==z_ops[i][1])],  axis=0)[1]
        sx_ops[i,1] = np.delete(s_stabs[i], [np.where(s_stabs[i]==x_ops[i][0]), \
				np.where(s_stabs[i]==x_ops[i][1])],  axis=0)[0]
        sz_ops[i,1] = np.delete(s_stabs[i], [np.where(s_stabs[i]==z_ops[i][0]), \
				np.where(s_stabs[i]==z_ops[i][1])],  axis=0)[0]
    return link_ops, x_ops, z_ops, s_stabs, sx_ops, sz_ops, logic_x, logic_z, logic_y, logic_s

def x_stabilizers(d):
    """
    make x-stabilizers for decoding with PyMatching
    x-stabilizers n dimensional bitstring, 1 where the stabilizer act with X and 0 where act with identity
    """
    x_stabs=np.zeros((d*(d-1), d**2+(d-1)**2), dtype=int)
    for i in range(d-1):
        x_stabs[i, i]=1
        x_stabs[i, i+1]=1
        x_stabs[i, i+d]=1
    for i in range(d-2):
        for j in range(d-1):
            x_stabs[(i+1)*(d-1)+j, (2*d-1)*(i+1)+j]=1
            x_stabs[(i+1)*(d-1)+j, (2*d-1)*(i+1)+j+1]=1
            x_stabs[(i+1)*(d-1)+j, (2*d-1)*(i+1)+j+d]=1
            x_stabs[(i+1)*(d-1)+j, (2*d-1)*(i+1)+j+1-d]=1
    for i in range(d-1):
        x_stabs[(d-1)**2+i, (2*d-1)*(d-1)+i]=1
        x_stabs[(d-1)**2+i, (2*d-1)*(d-1)+i+1]=1
        x_stabs[(d-1)**2+i, (2*d-1)*(d-1)+i+1-d]=1
    return x_stabs


def rotation(majoranas, M1, link_ops, link, phi):
    """
    Optimal algorithm for operation e^(-phi*link)
    tracking 3 quantities: active majorana modes, active submatrix of covariance matrix M1, link operators in inactive region
    link: 2 Majorana fermions on which rotation is done
    link1 and link2: MF's that are connected to the MF's in "link" by other link operators
    """
    new_majoranas=majoranas.copy()
    oldn_majoranas = len(new_majoranas)
    link1_in_majoranas = True       # If no external MF connected to link[0]
    #Add majoranas connected to link
    try:
        link1 = link_ops[(link_ops[:, 0]==link[0]) | (link_ops[:,1]==link[0])][0]   
        if link1[0] not in new_majoranas:
            new_majoranas.append(link1[0])
            link1_in_majoranas = False
        if link1[1] not in new_majoranas:
            new_majoranas.append(link1[1])
            link1_in_majoranas = False
    except:
        if link[0] not in new_majoranas:
            new_majoranas.append(link[0])
    link2_in_majoranas = True      # If no external MF connected to link[1]
    try:
        link2 = link_ops[(link_ops[:,0]==link[1]) | (link_ops[:,1]==link[1])][0]
        if link2[0] not in new_majoranas:
            new_majoranas.append(link2[0])
            link2_in_majoranas = False
        if link2[1] not in new_majoranas:
            new_majoranas.append(link2[1])
            link2_in_majoranas = False
    except:
        if link[1] not in new_majoranas:
            new_majoranas.append(link[1])
    newn_majoranas = len(new_majoranas)

    #construct active submatrix : 
    newM1 = np.zeros((newn_majoranas, newn_majoranas))
    newM1[:oldn_majoranas, :oldn_majoranas] = M1
    if link1_in_majoranas == False:
        newM1[new_majoranas.index(link1[0]), new_majoranas.index(link1[1])] = 1
        newM1[new_majoranas.index(link1[1]), new_majoranas.index(link1[0])] =-1
    if link2_in_majoranas == False:
        newM1[new_majoranas.index(link2[0]), new_majoranas.index(link2[1])] = 1
        newM1[new_majoranas.index(link2[1]), new_majoranas.index(link2[0])] =-1
    #indices of majorana fermions in link :
    p=new_majoranas.index(link[0])
    q=new_majoranas.index(link[1])

    #Only works for antisymmetric matricies
    newM2=newM1.copy()
    newM2[p, :]=newM1[p, :]*np.cos(2*phi)-newM1[q, :]*np.sin(2*phi)
    newM2[q, :]=newM1[q, :]*np.cos(2*phi)+newM1[p, :]*np.sin(2*phi)
    newM2[:, p]=newM1[:, p]*np.cos(2*phi)-newM1[:, q]*np.sin(2*phi)
    newM2[:, q]=newM1[:, q]*np.cos(2*phi)+newM1[:, p]*np.sin(2*phi)
    newM2[p, p]=0
    newM2[q, q]=0
    newM2[p, q]=newM1[p, q]*(np.cos(2*phi))**2-newM1[q, p]*(np.sin(2*phi))**2
    newM2[q, p]=-newM1[p, q]*(np.cos(2*phi))**2+newM1[q, p]*(np.sin(2*phi))**2
    return new_majoranas, newM2, link_ops


def measure_link(majoranas, M1, link_ops, link):
    """
    Optimal version of a measurement
    link: pair of indices, MF's that will be projected to "occupied"
    tracking 3 quantities: 
    majoranas = active majorana modes, 
    M1 = active submatrix of covariance matrix, 
    link_ops = link operators in inactive region
    """ 
    new_majoranas=majoranas.copy()
    oldn_majoranas    = len(new_majoranas)
    newlink_ops = link_ops
    link1_in_majoranas = True
    try:
        link1 = link_ops[(link_ops[:,0]==link[0]) | (link_ops[:,1]==link[0])][0]
        if link1[0] not in new_majoranas:
            new_majoranas.append(link1[0])
            link1_in_majoranas = False
        if link1[1] not in new_majoranas:
            new_majoranas.append(link1[1])
            link1_in_majoranas = False
        newlink_ops = np.delete(newlink_ops, np.where(newlink_ops == link1)[0], axis=0)
    except:
        if link[0] not in new_majoranas:
            new_majoranas.append(link[0])
    link2_in_majoranas = True
    try:
        link2 = link_ops[(link_ops[:,0]==link[1]) | (link_ops[:,1]==link[1])][0]
        if link2[0] not in new_majoranas:
            new_majoranas.append(link2[0])
            link2_in_majoranas = False
        if link2[1] not in new_majoranas:
            new_majoranas.append(link2[1])
            link2_in_majoranas = False
        newlink_ops = np.delete(newlink_ops, np.where(newlink_ops==link2)[0], axis=0)
    except:
        if link[1] not in new_majoranas:
            new_majoranas.append(link[1])
    
    #Construct active submatrix :
    newn_majoranas = len(new_majoranas)
    newM1 = np.zeros((newn_majoranas, newn_majoranas))
    newM1[:oldn_majoranas, :oldn_majoranas] = M1
    if link1_in_majoranas == False:
        newM1[new_majoranas.index(link1[0]), new_majoranas.index(link1[1])] = 1
        newM1[new_majoranas.index(link1[1]), new_majoranas.index(link1[0])] =-1
    if link2_in_majoranas == False:
        newM1[new_majoranas.index(link2[0]), new_majoranas.index(link2[1])] = 1
        newM1[new_majoranas.index(link2[1]), new_majoranas.index(link2[0])] =-1
    #indices of majorana fermions in link :
    p=new_majoranas.index(link[0])
    q=new_majoranas.index(link[1])

    # Measurement starts here:
    prob = 0.5*( 1 + newM1[p, q] )
    K = newM1[p, :]
    L = newM1[q, :]
    newM1 = newM1 + (np.outer(L,K)-np.outer(K,L)) / (2*prob)
    newM1 = np.delete(newM1, [p, q],0)
    newM1 = np.delete(newM1, [p, q],1)

    #remove measured link from majoranas :
    new_majoranas.remove(link[0])
    new_majoranas.remove(link[1])
    return prob, newM1, new_majoranas, newlink_ops



def syndrome_probability(n, syndrome, link_ops, x_ops, sx_ops, z_ops, phi):
    """
    optimal algorithm for calculating the probability of a given syndrome "m"
    """
    probs     = np.zeros(2*n)
    M1        = np.zeros((0, 0))
    majoranas = []
    for i in range(n):
        majoranas, M1, link_ops               = rotation(     majoranas, M1, link_ops, z_ops[i], phi[i]      )
        probs[2*i],   M1, majoranas, link_ops = measure_link( majoranas, M1, link_ops, x_ops[i,::int(syndrome[i])] )
        probs[2*i+1], M1, majoranas, link_ops = measure_link( majoranas, M1, link_ops, sx_ops[i,::int(syndrome[i])])
    return probs



def syndrome_logic_angle(d, phi, syndrome, pymatching_code, z_ops, x_ops, sx_ops, link_ops):
    """
    calculate logical rotation angle of a given "s" syndrome
    optimal algorithm
    """
    n=d**2+(d-1)**2
    logic_z = np.zeros(n)
    logic_z[:d] = 1
    syndrome_plus = np.ones(n)
    syndrome2     = -(syndrome-1)/2    # transform syndrome to 1/0 useful for pymatching
    correction_mwpm    = pymatching_code.decode(syndrome2) #correction string from mwpm decoder
    #only works for homogenous phis
    #calculation of error class likelihoods with mld from the inital error sttring suggested by mwpm
    log_P_I       = simulation_mld(np.sin(phi[0])**2, d, error_converter(d,correction_mwpm))
    log_P_Z       = simulation_mld(np.sin(phi[0])**2, d, error_converter(d,(correction_mwpm+logic_z)%2))
    if log_P_Z>log_P_I: # use the correction suggested by mld
        correction=(correction_mwpm+logic_z)%2
    else:
        correction=correction_mwpm
    phiplus       = phi + correction*np.pi/2
    phiminus      = phi + correction*np.pi/2 + logic_z*np.pi/2
    probsplus     = syndrome_probability(n, syndrome_plus, link_ops, x_ops, sx_ops, z_ops, phiplus)
    probsminus    = syndrome_probability(n, syndrome_plus, link_ops, x_ops, sx_ops, z_ops, phiminus)
    return np.prod(2*probsminus)/np.prod(2*probsplus)



def monte_carlo(n, link_ops, phi, z_ops, x_ops, sx_ops):
    """ 
    n: number of qubits
    link_ops: operators stabilizing the initial state
    phi: Z-rotations applied to physical qubits

    """
    msyndrome = np.zeros(n)     # for each qubit, X measurement outcomes
    prob      = np.zeros(n)     # for each measurement, probability of given outcome
    M1        = np.zeros((0, 0))# nontrivial part of covariance matrix, to be tracked
    majoranas = []     # array of Majoranas that M1 includes

    # First rotation and measurements on first qubit:
    majoranas, M1, link_ops     = rotation(    majoranas, M1, link_ops, z_ops[0], phi[0])
    
    # Measurement:
    if np.random.rand() < 0.5:   outcome = 1        
    else:                        outcome = -1
    p1, M1, majoranas, link_ops = measure_link(majoranas, M1, link_ops, x_ops[0,::outcome])
    
    p2, M1, majoranas, link_ops = measure_link(majoranas, M1, link_ops, sx_ops[0,::outcome])
    prob[0] = 2*p1*p2 #v
    msyndrome[0] = outcome       # keep track of the measurement result here

    # Rotations and measurements on other qubits:
    for i in range(1, n-1):
        # Rotation:
        majoranas, M1, link_ops = rotation(majoranas, M1, link_ops, z_ops[i], phi[i])
        # Measurement: 
        p1, newM1, newmajoranas, newlink_ops = measure_link(majoranas, M1, link_ops, x_ops[i])
        p2, newM1, newmajoranas, newlink_ops = measure_link(newmajoranas, newM1, newlink_ops, sx_ops[i])
        prob[i] = 2*p1*p2
        if np.random.rand() < prob[i]:
            M1, majoranas, link_ops = newM1, newmajoranas, newlink_ops
            msyndrome[i] = 1
        else:
            p1, M1,majoranas, link_ops = measure_link(majoranas, M1, link_ops, x_ops[i][::-1])
            p2, M1,majoranas, link_ops = measure_link(majoranas, M1, link_ops, sx_ops[i][::-1])
            prob[i]=2*p1*p2
            msyndrome[i]=-1
    # Rotation on last qubit:
    majoranas, M1, link_ops     = rotation(    majoranas, M1, link_ops, z_ops[n-1], phi[n-1])

    # Measurement on last qubit -- this is special: 
    p1, newM1, newmajoranas, newlink_ops = measure_link(majoranas, M1, link_ops, x_ops[n-1])
    p2, newM1, newmajoranas, newlink_ops = measure_link(newmajoranas, newM1, newlink_ops, sx_ops[n-1])
    prob[n-1] = p1*p2    # no factor of 2 here, because last stabilizer fixed due to fermion parity conservation
    if np.random.rand()<prob[n-1]:
        M1, majoranas, link_ops = newM1, newmajoranas, newlink_ops
        msyndrome[n-1] = 1
    else:
        p1, M1, majoranas, link_ops = measure_link(majoranas, M1, link_ops, x_ops[n-1][::-1])
        p2, M1, majoranas, link_ops = measure_link(majoranas, M1, link_ops, sx_ops[n-1][::-1])
        prob[n-1]      = p1*p2
        msyndrome[n-1] = -1
    return msyndrome, prob


#calculate attached "s" syndrome from a given "m" syndrome
def msyndrome_to_ssyndrome(d,m,x_stabs):
    ssyndrome=np.zeros(d*(d-1))
    for i in range(d*(d-1)):   ssyndrome[i] = np.prod(m[x_stabs[i]==1])
    return ssyndrome   

def error_converter(d,E):
    """
    Convert error string to form which is more natural for the mld algorithm
    """
    f=np.zeros((2,d,d))
    for i in range(d-1):
        for j in range(d):
            f[0,j,i] = E[i*(2*d-1)+j]
        for j in range(d-1):
            f[1,j,i] = E[i*(2*d-1)+d+j]
    for j in range(d):
        f[0,j,d-1] = E[(d-1)*(2*d-1)+j]
    return f

#Optimal algorithm for maximum likmelihood decoding
def initialize_M0(d):
    """
    Creation of the initial covariance metrix M0. This is the covariance matrix of the initial gaussian state |psi_e>.
    """
    M0=np.zeros((2*d,2*d))
    for i in range(d-1):
        M0[2*i+1,2*i+2]=1
        M0[2*i+2,2*i+1]=-1
    M0[0,2*d-1]=1
    M0[2*d-1,0]=-1
    return M0

def simulate_H_columns(M,j,log_gamma,f,p,d):
    """
    Algorithm for simulating the action of Gaussian operator H_j.
    """
    w=np.zeros(d)
    t=np.zeros(d)
    s=np.zeros(d)
    A=np.zeros((2*d,2*d))
    B=np.zeros((2*d,2*d))
    for i in range(d):
        if f[0,j,i]==1:
            w[i]=(1-p)/p
        else:
            w[i]=p/(1-p)
        log_gamma=log_gamma+np.log((1+w[i]**2)/2)
        t[i]=(1-w[i]**2)/(1+w[i]**2)
        s[i]=2*w[i]/(1+w[i]**2)
        A[2*i,2*i+1]=t[i]
        A[2*i+1,2*i]=-t[i]
        B[2*i,2*i]=s[i]
        B[2*i+1,2*i+1]=s[i]
    log_gamma=log_gamma+np.log(np.sqrt(np.linalg.det(M+A)))
    log_gamma=log_gamma+np.log(((1-p)**(d-np.sum(f[0,j,:]))*p**np.sum(f[0,j,:]))**2)
    M=A-(B@np.linalg.inv(M+A)@B)
    return M,log_gamma

def simulate_V_columns(M,j,log_gamma,f,p,d):
    """
    Algorithm for simulating the action of Gaussian operator V_j.
    """
    w=np.zeros(d-1)
    t=np.zeros(d-1)
    s=np.zeros(d-1)
    A=np.zeros((2*d,2*d))
    B=np.zeros((2*d,2*d))
    B[0,0]=1
    B[2*d-1,2*d-1]=1
    for i in range(d-1):
        if f[1,j,i]==1:
            w[i]=(1-p)/p
        else:
            w[i]=p/(1-p)
        log_gamma=log_gamma+np.log(1+w[i]**2)
        t[i]=2*w[i]/(1+w[i]**2)
        s[i]=(1-w[i]**2)/(1+w[i]**2)
        A[2*i+1,2*i+2]=t[i]
        A[2*i+2,2*i+1]=-t[i]
        B[2*i+1,2*i+1]=s[i]
        B[2*i+2,2*i+2]=s[i]
    log_gamma=log_gamma+np.log(np.sqrt(np.linalg.det(M+A)))
    log_gamma=log_gamma+np.log(((1-p)**(d-1-np.sum(f[1,j,:]))*p**np.sum(f[1,j,:]))**2)
    M=A-(B@np.linalg.inv(M+A)@B)
    return M,log_gamma

def simulation_mld(p,d,f):
    """
    Calculation of ln(Z({w_E})) from an inital error string, and the probability of errors.
    """
    M0=initialize_M0(d)
    M=M0.copy()
    log_gamma=np.log(2**(d-1))
    for j in range(d-1):
        M, log_gamma=simulate_H_columns(M,j,log_gamma,f,p,d)
        M, log_gamma=simulate_V_columns(M,j,log_gamma,f,p,d)  
    M, log_gamma=simulate_H_columns(M,d-1,log_gamma,f,p,d)
    return 1/2*log_gamma-1/2*np.log(2)+np.log(np.linalg.det(M+M0)**(1/4))

#parameters of the simulation
seed = 1
d    = 27
sim  = 10
phys_theta = 0.1 #in pi units

#parameters from command line
nparams = int((len(sys.argv)-1)/2)
for nparam in range(nparams):
    if sys.argv[1+nparam*2] == "-seed":
        print("# random seed: ", sys.argv[2+nparam*2])
        seed = int(sys.argv[2+nparam*2])
    if sys.argv[1+nparam*2] == "-d":
        print("# code distance: ", sys.argv[2+nparam*2])
        d = int(sys.argv[2+nparam*2])
    if sys.argv[1+nparam*2] == "-sim":
        print("# number of simulations: ", sys.argv[2+nparam*2])
        sim = int(sys.argv[2+nparam*2])
    if sys.argv[1+nparam*2] == "-theta":
        print("# physical rotation angle: ", sys.argv[2+nparam*2],'pi')
        phys_theta = float(sys.argv[2+nparam*2])

#logical angle distribution
np.random.seed(seed)
t0 = time.time()
n  = d**2+(d-1)**2
x_stabs = x_stabilizers(d)
pymatching_code = pm.Matching(x_stabs)

#majorana network:

link_ops, x_ops, z_ops, s_stabs, sx_ops, sz_ops, logic_x, logic_z, logic_y, logic_s = build_majorana_network(d)
t1 = time.time()

#stabilizer links for + state
logic_sx   = np.flip(logic_s.copy())
logic_sx   = np.delete(logic_sx, np.where(logic_sx == logic_x[0]), axis=0)
logic_sx   = np.delete(logic_sx, np.where(logic_sx == logic_x[1]), axis=0)
link_ops   = np.delete(link_ops, np.where(link_ops[:, 0] == 0),    axis=0)  # remove false links from link_ops
stabilizer_links = np.append(link_ops,          [logic_x], axis=0)
stabilizer_links = np.append(stabilizer_links, [logic_sx], axis=0)

#stabilizer links for y state
logic_sy = logic_s.copy()
logic_sy = np.delete(logic_sy, np.where(logic_sy == logic_y[0]), axis=0)
logic_sy = np.delete(logic_sy, np.where(logic_sy == logic_y[1]), axis=0)
stabilizer_links_y = np.append(link_ops, [logic_y], axis=0)
stabilizer_links_y = np.append(stabilizer_links_y, [logic_sy], axis=0)

phi    = np.zeros(n)
phi[:] = phys_theta*np.pi
for i in range(sim):
    t0 = time.time()
    m, prob = monte_carlo(n, stabilizer_links, phi, z_ops, x_ops, sx_ops)
    s     = msyndrome_to_ssyndrome(d, m, x_stabs)
    alpha = syndrome_logic_angle( d, phi,s, pymatching_code, z_ops, x_ops, sx_ops, stabilizer_links  )
    beta  = syndrome_logic_angle( d, phi,s, pymatching_code, z_ops, x_ops, sx_ops, stabilizer_links_y)
    t1 = time.time()
    print("%d %.5g %.5g %.5g "%(i, alpha, beta, t1-t0))
