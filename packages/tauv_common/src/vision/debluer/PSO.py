import numpy as np

#particle class
class Particle:
  def __init__(self, func, dim, vmin, vmax, seed):
    self.rnd = np.random.seed(seed)

    # initialize position, velocity, local_best_particle of the particle with 0.0 value
    self.velocity = np.zeros(dim)
    self.best_part_pos = np.zeros(dim)

    self.position = np.random.uniform(vmin, vmax, dim)
 
    # compute fitness of particle
    self.fitness = func(self.position) # curr fitness
 
    # initialize best position and fitness of this particle
    self.best_part_pos = np.copy(self.position)
    self.best_part_fitness = self.fitness     # best fitness

 
def pso(func, max_iter, num_particles, dim, vmin, vmax, params):

  # hyper parameters
  wmax = params["wmax"]    # maximum inertia
  wmin = params["wmin"]    #minimum inertia
  c1 = params["c1"] 	   # cognitive (particle)
  c2 = params["c2"]       # social (swarm)
 
  rnd = np.random.seed()
 
  # create num_particles
  swarm = [Particle(func, dim, vmin, vmax, i) for i in range(num_particles)]
 
  # compute the value of best_position and best_fitness in swarm
  best_swarm_pos = np.zeros(dim)
  best_swarm_fitness = np.inf # swarm best
 
  # computer best particle of swarm and it's fitness
  for i in range(num_particles): # check each particle
    if swarm[i].fitness < best_swarm_fitness:
      best_swarm_fitness = swarm[i].fitness
      best_swarm_pos = np.copy(swarm[i].position)
 
  # main loop of pso
  it = 0
  while it < max_iter:
     
    # For every 5 iterations print iteration number and best fitness value
    if it % 5 == 0:
      print("Iteration = " + str(it) + " best fitness = %f" % best_swarm_fitness)
    
    w = wmax - ((wmax - wmin)/max_iter)*it

    for i in range(num_particles): 
       
      # compute new velocity of current particle
      swarm[i].velocity = (
                           (w * swarm[i].velocity) +
                           (c1 * np.random.rand(dim) * (swarm[i].best_part_pos - swarm[i].position)) + 
                           (c2 * np.random.rand(dim) * (best_swarm_pos -swarm[i].position))
                         ) 

      # compute new position using new velocity
      for k in range(dim):
        swarm[i].position[k] += swarm[i].velocity[k]
        swarm[i].position[k] = np.maximum(swarm[i].position[k], vmin)
        swarm[i].position[k] = np.minimum(swarm[i].position[k], vmax)

      # compute fitness of new position
      swarm[i].fitness = func(swarm[i].position)
 
      # check for local best particle
      if swarm[i].fitness < swarm[i].best_part_fitness:
        swarm[i].best_part_fitness = swarm[i].fitness
        swarm[i].best_part_pos = np.copy(swarm[i].position)
 
      # check for global best particle
      if swarm[i].fitness < best_swarm_fitness:
        best_swarm_fitness = swarm[i].fitness
        best_swarm_pos = np.copy(swarm[i].position)

    it += 1
  
  gbest ={}
  gbest["position"] = best_swarm_pos
  gbest["cost"] = best_swarm_fitness

  return gbest