'''
author: Alp Oran
'''

import numpy as np

# swarm parameters
numDummiesP = 10
numDummiesZ = 5
numParticles = 40
kNearest = 6

# physics parameters
kg, kf, hPhysics = 2.5, 3, 0.1
hGrad = 1e-4            # step size
nudgeFactor = 0.05      # amount to shake after finding a pole
gDummyAttraction = 1.2  # dummies attraction to the average
gDummySmell = 1.0       # dummies attraction to the poles 
kfDummy = 5.0           # dummy friction (allows for it to lag behind a bit)

# creates the random scatter of scouts
np.random.seed(42)
pP = np.random.uniform(-12, 12, (numParticles, 1)) + 1j * np.random.uniform(-12, 12, (numParticles, 1))
vP = np.zeros(pP.shape, dtype=np.complex128)
dP = np.random.uniform(-10, 10, (numDummiesP, 1)) + 1j * np.random.uniform(-10, 10, (numDummiesP, 1))
vDP = np.zeros(dP.shape, dtype=np.complex128)
poles = []

pZ = np.random.uniform(-12, 12, (numParticles, 1)) + 1j * np.random.uniform(-12, 12, (numParticles, 1))
vZ = np.zeros(pZ.shape, dtype=np.complex128)
dZ = np.random.uniform(-10, 10, (numDummiesZ, 1)) + 1j * np.random.uniform(-10, 10, (numDummiesZ, 1))
vDZ = np.zeros(dZ.shape, dtype=np.complex128)
zeros = []

# defines the function
def N(s):
    return s - 4

def D(s): 
    return (s**2 + 6*s + 25) * (s**2 - 12*s + 40) * (s + 10)

# use to optimize for minimal cost
def cost(s, foundList, func):
    val = np.log10(np.abs(func(s)) + 1e-15)
    for f in foundList:
        val -= 2.0 * np.log10(np.abs(s - f) + 1e-15)
    return val

# use for gradient descent
def gradCost(pIn, foundList, func):
    h = hGrad
    pArr = np.array(pIn).reshape(-1, 1)
    delX = (cost(pArr + h, foundList, func) - cost(pArr - h, foundList, func)) / (2*h)
    delY = (cost(pArr + 1j*h, foundList, func) - cost(pArr - 1j*h, foundList, func)) / (2*h)
    g = delX + 1j * delY
    mag = np.abs(g)
    return np.where(mag > 1e-9, g / mag, 0)

# iteration loop of 2nd order gradient descent
i = 0
while i < 30000:  #seemed to converge way before hitting this number for a 5th order equation
    gPP = gradCost(pP, poles, D) # the gradient
    vP = (vP + (-gPP * kg) * hPhysics) * np.exp(-hPhysics * kf) # calculates velocity/momentum
    pP = pP + (vP * hPhysics) + (gPP * nudgeFactor) # based on velocity computes position
    pP.real, pP.imag = np.clip(pP.real, -12, 12), np.clip(pP.imag, -12, 12)
    
    # dummy physics
    if len(dP) > 0:
        distDP = np.abs(dP - pP.T)
        accDP = np.zeros(dP.shape, dtype=np.complex128)
        for dIdx in range(len(dP)):
            idx = np.argsort(distDP[dIdx])[:kNearest]
            fAttract = (np.mean(pP[idx]) - dP[dIdx]) * gDummyAttraction #this is the dummies attraction to the average
            fSmell = -gradCost(dP[dIdx], poles, D) * gDummySmell        #this is the dummies attraction to the poles
            accDP[dIdx] = fAttract + fSmell         #the combined attraction allows for the dummy to both follow the gradient but also quickly converge to the poles
        vDP = (vDP + accDP * hPhysics) * np.exp(-hPhysics * kfDummy) #same sort of logic I used previously
        dP = dP + (vDP * hPhysics)
        #sees if we found a root
        toDelP = []
        for dIdx in range(len(dP)):
            cand = dP[dIdx][0]
            if np.abs(D(cand)) < 0.2:
                if not any(np.abs(cand - pf) < 0.6 for pf in poles):
                    poles.append(cand)
                    print(f"Iter {i}: Pole Scout {dIdx} found {cand:.4f}")
                    toDelP.append(dIdx)
                    vP += np.random.normal(0, 5, pP.shape) + 1j*np.random.normal(0, 5, pP.shape)
        if toDelP:
            dP = np.delete(dP, toDelP, axis=0)
            vDP = np.delete(vDP, toDelP, axis=0)

    gPZ = gradCost(pZ, zeros, N)
    vZ = (vZ + (-gPZ * kg) * hPhysics) * np.exp(-hPhysics * kf)
    pZ = pZ + (vZ * hPhysics) + (gPZ * nudgeFactor) #shakes up the board to make sure we are at pole
    pZ.real, pZ.imag = np.clip(pZ.real, -12, 12), np.clip(pZ.imag, -12, 12)
    # same logic as the poles applied to zeros
    if len(dZ) > 0:
        distDZ = np.abs(dZ - pZ.T)
        accDZ = np.zeros(dZ.shape, dtype=np.complex128)
        for dIdx in range(len(dZ)):
            idx = np.argsort(distDZ[dIdx])[:kNearest]
            fAttract = (np.mean(pZ[idx]) - dZ[dIdx]) * gDummyAttraction
            fSmell = -gradCost(dZ[dIdx], zeros, N) * gDummySmell
            accDZ[dIdx] = fAttract + fSmell
        vDZ = (vDZ + accDZ * hPhysics) * np.exp(-hPhysics * kfDummy)
        dZ = dZ + (vDZ * hPhysics)
        
        toDelZ = []
        for dIdx in range(len(dZ)):
            cand = dZ[dIdx][0]
            if np.abs(N(cand)) < 0.2:
                if not any(np.abs(cand - zf) < 0.6 for zf in zeros):
                    zeros.append(cand)
                    print(f"Iter {i}: Zero Scout {dIdx} found {cand:.4f}")
                    toDelZ.append(dIdx)
                    vZ += np.random.normal(0, 5, pZ.shape) + 1j*np.random.normal(0, 5, pZ.shape)
        if toDelZ:
            dZ = np.delete(dZ, toDelZ, axis=0)
            vDZ = np.delete(vDZ, toDelZ, axis=0)

    if len(poles) >= 5 and len(zeros) >= 1:
        break
    i += 1
#this checks for any duplicates, it is quite simple, it just checks when it becomes zero with a little tolerance
def checkDupes(root, func, tol=1e-3):
    dupeCount = 1
    curFunc = func
    while True:
        tVal = root + 1e-6 
        rVal = curFunc(tVal) / (tVal - root)
        if abs(rVal) < tol:
            dupeCount += 1
            tempFunc = curFunc # scope capture
            curFunc = lambda s, tf=tempFunc, r=root: tf(s) / (s - r)
        else:
            break
    return dupeCount

uniquePoles = []
for p in poles:
    if not any(abs(p - up) < 0.1 for up in uniquePoles):
        uniquePoles.append(p)

uniqueZeros = []
for z in zeros:
    if not any(abs(z - uz) < 0.1 for uz in uniqueZeros):
        uniqueZeros.append(z)

finalPoles = []
finalZeros = []

print("\n" + "="*30)
print("   S-PLANE SYSTEM ANALYSIS")
print("="*30)

print("\n--- POLES (Denominator Roots) ---")
for root in uniquePoles:
    count = checkDupes(root, D)
    print(f"Location: {root.real:+.4f} {root.imag:+.4f}j | Multiplicity: {count}")
    for _ in range(count):
        finalPoles.append(root)

print("\n--- ZEROS (Numerator Roots) ---")
for root in uniqueZeros:
    count = checkDupes(root, N)
    print(f"Location: {root.real:+.4f} {root.imag:+.4f}j | Multiplicity: {count}")
    for _ in range(count):
        finalZeros.append(root)

print("\n" + "="*30)
print(f"Summary: {len(finalPoles)} Poles, {len(finalZeros)} Zeros found.")
print("="*30)
