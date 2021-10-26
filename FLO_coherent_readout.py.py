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
    Build up the proper Majorana network of a rotated surface code with code distance d:
    Integer index of each Majorana is defined as: 
    starting from bottom left, go up and then switch columns, index*4. Then add integer index of each 
    Majorana: 0, 1, 2, 3 starting from bottom, going clockwise.
    S stabilizers, X and Z operators therefore are defined differently on even and odd qubits.
    """
    s_stabs   = np.zeros((d*d, 4), int) #stabilizers of c4 codes (qubits, -c1c2c3c4)
    z_ops     = np.zeros((d*d, 2), int) #z-operators of c4 codes (ic2c3)
    x_ops     = np.zeros((d*d, 2), int) #x-operators of c4 codes (ic1c2)
    link_ops  = np.zeros(((d*d+1)*2, 2), int) #link operators 
    logic_x = np.array([0, 4*(d*d-d)+1], int)
    logic_z = np.array([4*(d*d-d)+1, 4*(d*d-1)+2], int)
    logic_y = np.array([4*(d*d-1)+2, 0], int)
    logic_s = np.array([0, 4*(d*d-d)+1, 4*(d*d-1)+2, 4*(d-1)+3], int)
    for i in range(d*d): #fill up the operators located on qubits
        if i%2 == 0:
            s_stabs[i] = [4*i,  4*i+1, 4*i+2, 4*i+3]
            x_ops[i]   = [4*i,  4*i+1]
            z_ops[i]   = [4*i+1, 4*i+2]
        else:
            s_stabs[i] = [4*i+1, 4*i+2, 4*i+3, 4*i]
            x_ops[i]   = [4*i+1, 4*i+2]
            z_ops[i]   = [4*i+2, 4*i+3]
    for i in range((d*d+1)//2): #fill up the link operators, its a bit tricky
        n = 2*i
        if n == 0:
            link_ops[4*i+1] = [4*n+1,   4*(n+d)+1]
            link_ops[4*i+2] = [4*(n+d), 4*n+2] 
            link_ops[4*i+3] = [4*n+3,   4*(n+1)+1]
        elif n==(d-1):
            link_ops[4*i]   = [4*(n-1),   4*n  ]
            link_ops[4*i+1] = [4*(n-1)+3, 4*n+1]
            link_ops[4*i+2] = [4*(n+d),   4*n+2]
        elif n==(d*d-d):
            link_ops[4*i]   = [4*(n-d)+2, 4*n  ]
            link_ops[4*i+2] = [4*n+2,     4*(n+1)+2]
            link_ops[4*i+3] = [4*n+3,     4*(n+1)+1]
        elif n==(d*d-1):
            link_ops[4*i]   = [4*(n-d)+2, 4*n]
            link_ops[4*i+1] = [4*(n-1)+3, 4*n+1]
            link_ops[4*i+3] = [4*n+3,     4*(n-d)+3]
        else:
            if n<=(d-1):
                link_ops[4*i]   = [4*(n-1),  4*n]
                link_ops[4*i+1] = [4*(n-1)+3,4*n+1]
                link_ops[4*i+2] = [4*(n+d),  4*n+2]
                link_ops[4*i+3] = [4*n+3,    4*(n+1)+1]
            elif n>(d-1) and n%d!=0 and n%d!=(d-1) and n<(d*d-d):
                if n//d%2==0:
                    link_ops[4*i]   = [4*(n-d)+2, 4*n]
                    link_ops[4*i+1] = [4*(n-1)+3, 4*n+1]
                    link_ops[4*i+2] = [4*(n+d),   4*n+2]
                    link_ops[4*i+3] = [4*n+3,     4*(n+1)+1]
                else:
                    link_ops[4*i]   = [4*n,       4*(n-d)+2]
                    link_ops[4*i+1] = [4*n+1,     4*(n-1)+3]
                    link_ops[4*i+2] = [4*n+2,     4*(n+d)]
                    link_ops[4*i+3] = [4*(n+1)+1, 4*n+3]
            elif n%d==0:
                link_ops[4*i]   = [4*(n-d)+2, 4*n]
                link_ops[4*i+1] = [4*n+1,     4*(n+d)+1]
                link_ops[4*i+2] = [4*(n+d),   4*n+2]
                link_ops[4*i+3] = [4*n+3,     4*(n+1)+1]
            elif n%d==(d-1):
                link_ops[4*i]   = [4*(n-d)+2, 4*n]
                link_ops[4*i+1] = [4*(n-1)+3, 4*n+1]
                link_ops[4*i+2] = [4*(n+d),   4*n+2]
                link_ops[4*i+3] = [4*n+3,     4*(n-d)+3]
            elif n>(d*d-d):
                link_ops[4*i]   = [4*(n-d)+2, 4*n]
                link_ops[4*i+1] = [4*(n-1)+3, 4*n+1]
                link_ops[4*i+2] = [4*n+2,     4*(n+1)+2]
                link_ops[4*i+3] = [4*n+3,     4*(n+1)+1]
    sx_ops = np.zeros((d*d, 2), dtype=int) #s*x operators, ic3c4
    sz_ops = np.zeros((d*d, 2), dtype=int) #s*z operators, ic2c3
    for i in range(d*d):
        sx_ops[i] = np.delete(s_stabs[i], [np.where(s_stabs[i]==x_ops[i][0]), \
				np.where(s_stabs[i]==x_ops[i][1])],  axis=0)
        sz_ops[i] = np.delete(s_stabs[i], [np.where(s_stabs[i]==z_ops[i][0]), \
				np.where(s_stabs[i]==z_ops[i][1])],  axis=0)
    return link_ops, x_ops, z_ops, s_stabs, sx_ops, sz_ops, logic_x, logic_z, logic_y, logic_s

def x_stabilizers(d):
    """
    make x-stabilizers for decoding with PyMatching
    x-stabilizers n dimensional bitstring, 1 where the stabilizer act with X and 0 where act with identity
    """
    x_stabs=np.zeros(((d*d-1)//2, d*d), dtype=int)
    for i in range((d*d-1)//2):
        if i<(d-1)//2:
            x_stabs[i, 2*i+1:2*i+3] = [1, 1]
        elif i>=((d*d-1)//2-(d-1)//2):
            x_stabs[i, 2*i : 2*i+2]   = [1, 1]
        else:
            n=2*(i//(d-1))+1
            x_stabs[i, 2*i+n   : 2*i+n+2]    = [1, 1]
            x_stabs[i, 2*i+n-d : 2*i+n-d+2]  = [1, 1]
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

    #Apply rotation :
    """newM2=newM1.copy()
    R = np.eye(newn_majoranas)
    R[p, p] = np.cos(2*phi)
    R[q, q] = np.cos(2*phi)
    R[p, q] =-np.sin(2*phi)
    R[q, p] = np.sin(2*phi)
    newM2 = R @ newM2 @ R.T"""
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



def syndrome_logic_angle(n, phi, syndrome, pymatching_code, z_ops, x_ops, sx_ops, link_ops):
    """
    calculate logical rotation angle of a given "s" syndrome
    optimal algorithm
    """
    logic_z = np.zeros(n)
    logic_z[:int(np.sqrt(n))] = 1
    syndrome_plus = np.ones(n)
    syndrome2     = -(syndrome-1)/2    # transform syndrome to 1/0 useful for pymatching
    correction    = pymatching_code.decode(syndrome2) #correction string from mwpm decoder
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


def msyndrome_to_ssyndrome(d,m,x_stabs):
    """

    calculate attached "s" syndrome from a given "m" syndrome

    """
    ssyndrome=np.zeros((d*d-1) // 2)
    for i in range((d*d-1) // 2):   ssyndrome[i] = np.prod(m[x_stabs[i]==1])
    return ssyndrome   

def noisy_syndrome(d,p):
    s_noise=(np.random.rand(d**2)<p).astype(np.uint8)
    return s_noise
    
def simulation_round(n, stabilizer_links, stabilizer_links_y, x_ops, sx_ops, z_ops, x_stabs, pymatching_code, phi):
    """
    
    Simmulate one round of error correction. Returns a logical rotation angle sampled from p(s).
    
    """
    m, prob = monte_carlo(n, stabilizer_links, phi, z_ops, x_ops, sx_ops)
    s       = msyndrome_to_ssyndrome(d, m, x_stabs)
    alpha   = syndrome_logic_angle( n, phi,s, pymatching_code, z_ops, x_ops, sx_ops, stabilizer_links  )
    beta    = syndrome_logic_angle( n, phi,s, pymatching_code, z_ops, x_ops, sx_ops, stabilizer_links_y)
    theta   = np.arctan2( (1-beta)/(beta+1), (1-alpha)/(alpha+1) ) / 2.
    if theta<0:
        theta=np.pi+theta
    return theta,s

def noisy_measurement(n, stabilizer_links, stabilizer_links_y, x_ops, sx_ops, z_ops, x_stabs, pymatching_code, pymatching_code_3D, phi, n_sims):
    """
    
    Simulate d-round of stabilizer measurments and calculate correction string from 3D noisy syndrome. After the last round 
    "perfect measurment" round returns the code to the codespace.
    
    """
    d          = int(np.sqrt(n))
    phi_0      = phi.copy()
    theta      = np.zeros(d)
    correction = np.zeros((d , d**2 ))
    syndrome   = np.zeros(((d**2-1)//2 , d))
    syndrome_pymatch = np.zeros(((d**2-1)//2 , d))
    
    for i in range(d):
        modi_phi                = phi_0+correction[i-1,:]*(np.pi/2)
        theta[i], syndrome[:,i] = simulation_round(n, stabilizer_links, stabilizer_links_y, x_ops, sx_ops, z_ops, x_stabs, pymatching_code, modi_phi)
        syndrome_pymatch[:,i]   = -(syndrome[:,i]-1)/2
        correction[i,:]         = pymatching_code.decode(syndrome_pymatch[:,i])

    print(theta, file=sys.stderr)
    theta_star = np.sum(theta)
    theta_1    = theta_star.copy()%np.pi
    theta_2    = (theta_1+np.pi/2)%np.pi
    n_wrongs=0
    for i in range(n_sims):
        noise      = (np.random.rand((d**2-1)//2 , d) < np.sin(phi[0])**2).astype(np.uint8)
        noise[:,-1]= 0
        noisy_syndrome            = (syndrome_pymatch + noise) %2
        difference_syndrome       = noisy_syndrome.copy()
        difference_syndrome[:,1:] = (noisy_syndrome[:,1:]-noisy_syndrome[:,0:-1]) %2
        correction_3D             = pymatching_code_3D.decode(difference_syndrome)
        final_correction=(correction_3D + correction[-1 , :])%2
        if np.sum(final_correction) %2 == 1:
            n_wrongs+=1
        
        
    return theta_1, theta_2, n_wrongs
    


#parameters of the simulation
seed = 2
d    = 5
sim  = 50
phys_theta = 0.05 #in pi units
number_of_sims = 100

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
n  = d*d
x_stabs = x_stabilizers(d)
pymatching_code = pm.Matching(x_stabs)
pymatching_code_3D = pm.Matching(x_stabs , repetitions=d)

#majorana network:

link_ops, x_ops, z_ops, s_stabs, sx_ops, sz_ops, logic_x, logic_z, logic_y, logic_s = build_majorana_network(d)
t1 = time.time()

#stabilizer links for + state
logic_sx   = logic_s.copy()
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
    t0=time.time()
    L_theta, L_theta_error, sims=noisy_measurement(n, stabilizer_links, stabilizer_links_y, x_ops, sx_ops, z_ops, x_stabs, pymatching_code, pymatching_code_3D, phi, number_of_sims)
    t1=time.time()
    print("%d %.5g %.5g %d %.5g"%(i,L_theta, L_theta_error, sims, t1-t0))

