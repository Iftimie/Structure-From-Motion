import numpy as np
import cv2

#http://note.sonots.com/SciSoftware/Factorization.html
#https://sourceforge.net/p/cvprtoolbox/code/HEAD/tree/
#http://note.sonots.com/?plugin=attach&refer=SciSoftware%2FFactorization&openfile=Factorization.pdf
##http://cs.brown.edu/courses/cs143/2011/proj5/Tomasi_Kanade_IJCV_1992.pdf
#http://cs.brown.edu/courses/cs143/2011/proj5/
#https://sourceforge.net/p/cvprtoolbox/code/HEAD/tree/project/Factorization/cvFactorization.m#l110
#https://www-inst.cs.berkeley.edu/~cs294-6/fa06/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf
#http://note.sonots.com/?plugin=attach&refer=SciSoftware%2FFactorization&openfile=Factorization.pdf
#http://note.sonots.com/SciSoftware/Factorization.html


cap = cv2.VideoCapture('Brunei building rotation 09.02.14a.avi')
for i in range(60):
    ret, frame = cap.read()
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 300,
                       qualityLevel = 0.15,
                       minDistance = 20,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(300,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
maximumFrames = 48

allPoints = []
allPoints.append(p0)
x=0
while(1):
    x = x + 1
    if x>47:
        break

    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    allPoints.append(good_new)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    if x==1:
        k = cv2.waitKey(5000) & 0xff
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()

print (len(allPoints))
F = 48
P = 155
W = np.zeros((2*F,P), dtype=np.float64)

for p in range(P):
    value = allPoints[0][p][0][1]
    W[0, p] = allPoints[0][p][0][0]
    W[0 + F, p] = allPoints[0][p][0][1]
for f in range(1,F):
    for p in range(P):
        value = allPoints[f][p][1]
        W[f,p] = allPoints[f][p][0]
        W[f+F,p] = allPoints[f][p][1]




##########Normalize

normW = np.zeros((2*F,P), dtype=np.float64)

t = []
e = np.ones((1, P))

for f in range(F):
    af = 0
    for p in range(P):
        af = af + W[f,p]
    af = af/P
    t.append([af])
    for p in range(P):
        normW[f,p] = W[f,p]-af


    bf = 0
    for p in range(P):
        bf = af +W[f+F,p]
    bf = bf/P
    t.append([bf])
    for p in range(P):
        normW[f+F,p] = W[f+F,p]-bf


U, S, V = np.linalg.svd(normW,full_matrices=False)

O1 =U[:,:3]
S1 = np.diag(S)[:3,:3]
O2 = V[:3,:]

Westim = np.dot(np.dot(O1,S1),O2)

Restim = np.dot(O1,np.sqrt(S1))
Sestim = np.dot(np.sqrt(S1),O2)

G = np.zeros((3*F,6), dtype=np.float64)

def gT(a,b):
    gr = np.zeros((6,1), dtype=np.float64)
    gr[0,0] = a[0]*b[0]
    gr[1,0] = a[0]*b[1]+a[1]*b[0]
    gr[2,0] = a[0]*b[2]+a[2]*b[1]
    gr[3,0] = a[1]*b[1]
    gr[4,0] = a[1]*b[2]+a[2]*b[2]
    gr[5,0] = a[2]*b[2]
    return np.transpose(gr)

def orthometric(Rh): #2*Fx3
    ihT = Rh[:F,:] #Fx3
    jhT = Rh[F:2*F,:] #Fx3
    for i in range(F):
        G[i] = gT(ihT[i],ihT[i])
        G[i+F] = gT(jhT[i],jhT[i])
        G[i+2*F] = gT(ihT[i],jhT[i])

orthometric(Restim)

L = np.zeros((3,3), dtype=np.float64)

I = np.zeros((6,1), dtype=np.float64)

c = np.ones((3*F,1), dtype=np.float64)
for x in range (2*F,3*F):
    c[x,0] = 0

I = np.dot(np.linalg.pinv(G),c)

L = np.array([[I[0,0] , I[1,0], I[2,0]],[I[1,0],I[3,0],I[4,0]],[I[2,0],I[4,0],I[5,0]]])
D,V = np.linalg.eig(L)
D = np.diag(D)

D = np.where(D < 0,0.00001,D)
Q = np.dot(V,np.sqrt(D))

R = np.dot(Restim,Q)
S = np.dot(np.linalg.inv(Q),Sestim)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure('%d' % (0))
ax = fig.add_subplot(111, projection='3d')
xs1 = []
ys1 = []
zs1 = []
for x in range(P):
    xs1.append(S[0,x])
    ys1.append(S[1,x])
    zs1.append(S[2,x])
ax.ticklabel_format(useOffset=False)
ax.scatter(xs1, ys1, zs1, c='b', marker='o')  # c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]

plt.show()

i1 = np.transpose(R[1,:])
i1 = i1 / np.linalg.norm(i1)
j1 = np.transpose(R[F,:])
j1 = j1 /np.linalg.norm(i1)
k1 = np.cross(i1,j1)
k1  = k1 / np.linalg.norm(k1)
R0 = np.array([i1, j1, k1])
R = np.dot(R,R0)
S = np.dot(np.linalg.inv(R0),S) #3XP


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure('%d' % (0))
ax = fig.add_subplot(111, projection='3d')
xs1 = []
ys1 = []
zs1 = []
for x in range(P):
    xs1.append(S[0,x])
    ys1.append(S[1,x])
    zs1.append(S[2,x])
ax.ticklabel_format(useOffset=False)
ax.scatter(xs1, ys1, zs1, c='b', marker='o')  # c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]

plt.show()