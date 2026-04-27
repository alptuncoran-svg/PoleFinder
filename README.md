# PoleFinder
a code that solves for roots in the s plane using swarm particle optimization


  The code itself is pretty basic. It is a physics infromed pole/zero finder meaning the particles themselves feel inertia and friction unlike a standard gradient descent which allow for the particles to
escape local minima, however this introduces steady state error due to the fact that the particles just start orbiting the poles rather than converging, this is why friction is important because it acts as a damper which allows for convergence.
  There are also some particle swarm attributes in the form of "dummies" which are basically too dumb to think for themselves and do whatever the crowd around them does, this was achieved by having the dummies be attracted to the poles as well as the average point of the closest points. With this behavior the dummies are able to converge faster than the rest of the particles. 
    There is also a deflation logic where once a pole is find the cost function is updated in a way to blow it up around the found pole thus discouraging the particles to clump around a found pole.
    There also is a small sub routine that checks wheter a pole is a multiple pole which works by testing how many times we need to divide the function by (s-p) to reach a zero. 
    The code should be able to run on pretty much anything that can run python
