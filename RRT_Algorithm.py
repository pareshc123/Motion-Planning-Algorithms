# RRT algorithm
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 10


class TreeNode:

    def __init__(self, location_x, location_y):
        self.location_x = location_x         # x Location
        self.location_y = location_y         # y Location
        self.children = []                   # children list
        self.parent = None                   # parent node reference


# RRT Algorithm
class RRTAlgorithm:

    def __init__(self, start_pos, goal_pos, num_iterations, grid_size, step_size):
        self.root_node = TreeNode(start_pos[0], start_pos[1])      # The RRT (root position)
        self.goal = TreeNode(goal_pos[0], goal_pos[1])               # goal position
        self.nearestNode = None                                      # nearest node
        self.iterations = min(num_iterations, 250)                   # number of iterations to run
        self.grid = grid_size                                        # the map
        self.rho = step_size                                         # length of each branch
        self.path_distance = 0                               # total path distance
        self.nearestDist = 10000                             # distance to the nearest node (initialize with large)
        self.numWaypoints = 0                                # number of waypoints
        self.Waypoints = []                                  # the waypoints

    # add the node to the nearest node, and add goal if necessary
    def add_child(self, location_x, location_y):

        if location_x == self.goal.location_x:
            self.nearestNode.children.append(self.goal)          # append goal to nearestNode's children
            self.goal.parent = self.nearestNode                  # and set goal's parent to nearestNode
        else:
            current_node = TreeNode(location_x, location_y)      # create a tree node from location_x, location_y
            self.nearestNode.children.append(current_node)       # append this node to nearestNode's children
            current_node.parent = self.nearestNode               # set the parent to nearestNode

    # sample random point within grid limits
    @staticmethod
    def sample_new_point():
        x = random.randint(1, grid.shape[1])
        y = random.randint(1, grid.shape[0])
        new_point = np.array([x, y])
        return new_point

    # steer a distance stepSize from start location to end location
    def steer_to_new_point(self, location_start, location_end):
        offset = self.rho * self.unitVector(location_start, location_end)
        new_point = np.array([location_start.location_x + offset[0], location_start.location_y + offset[1]])
        if new_point[0] >= grid.shape[1]:
            new_point[0] = grid.shape[1] - 1
        if new_point[1] >= grid.shape[0]:
            new_point[1] = grid.shape[0] - 1
        return new_point

    # check if obstacle lies between the start and end point of the edge
    def is_in_obstacle(self, location_start, location_end):
        u_hat = self.unitVector(location_start, location_end)
        test_point = np.array([0.0, 0.0])
        for r in range(self.rho):
            test_point[0] = min(grid.shape[1] - 1, location_start.location_x + r * u_hat[0])
            test_point[1] = min(grid.shape[0] - 1, location_start.location_y + r * u_hat[1])
            if self.grid[round(test_point[1]), round(test_point[0])] == 1:
                return True
        return False

    @staticmethod
    # find the unit vector between 2 locations (DONE)
    def unitVector(location_start, location_end):
        v = np.array([location_end[0] - location_start.location_x, location_end[1] - location_start.location_y])
        u_hat = v / np.linalg.norm(v)
        return u_hat

    # find the nearest node from a given (unconnected) point (Euclidean distance)
    def findNearest(self, root_node, curr_point):
        if not root_node:
            return

        dist = self.distance(root_node, curr_point)        # find distance between root and point use distance method

        if dist <= self.nearestDist:             # update the node
            self.nearestNode = root_node
            self.nearestDist = dist

        for child in root_node.children:
            self.findNearest(child, curr_point)

    @staticmethod
    # find euclidean distance between a node object and an XY point
    def distance(node1, curr_point):
        dist = math.sqrt((node1.location_x - curr_point[0]) ** 2 + (node1.location_y - curr_point[1]) ** 2)
        return dist

    # check if the goal is within step size (rho) distance from point, return true if so otherwise false
    def goalFound(self, curr_point):
        if self.distance(self.goal, curr_point) <= self.rho:
            return True
        return False

    # reset: set nearestNode to None and nearestDistance to 10000
    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000

    # trace the path from goal to start
    def retraceRRTPath(self, final_goal):

        if final_goal.location_x == self.root_node.location_x:
            return
        self.numWaypoints += 1            # add 1 to numWaypoints

        # extract the X Y location of goal in a numpy array
        current_point = np.array([final_goal.location_x, final_goal.location_y])
        self.Waypoints.insert(0, current_point)                # insert this array to waypoints (from the beginning)
        self.path_distance += self.rho                         # add rho to path_distance

        self.retraceRRTPath(final_goal.parent)


# load the grid, set start and goal <x, y> positions, number of iterations, step size
grid = np.load('cspace.npy')
start = np.array([100.0, 100.0])
goal = np.array([1600.0, 750.0])
numIterations = 225
stepSize = 100
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

fig = plt.figure("RRT Algorithm")
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
plt.xlabel('X-axis $(m)$')
plt.ylabel('Y-axis $(m)$')

# Begin
rrt = RRTAlgorithm(start, goal, numIterations, grid, stepSize)
plt.pause(2)

# RRT algorithm
for i in range(rrt.iterations):

    # Reset the nearest values
    rrt.resetNearestValues()
    print("Iteration: ", i)

    # sample a point
    point = rrt.sample_new_point()
    rrt.findNearest(rrt.root_node, point)
    new = rrt.steer_to_new_point(rrt.nearestNode, point)  # steer to a point, return as 'new'

    # if not in obstacle
    if not rrt.is_in_obstacle(rrt.nearestNode, new):
        # add new to the nearest node (addChild), again no need to return just call the method
        rrt.add_child(new[0], new[1])
        plt.pause(0.10)
        plt.plot([rrt.nearestNode.location_x, new[0]], [rrt.nearestNode.location_y, new[1]], 'go', linestyle="--")

        # if goal found (new is within goal region)
        if rrt.goalFound(new):
            # append goal to path
            rrt.add_child(goal[0], goal[1])
            rrt.retraceRRTPath(rrt.goal)
            print("Goal Found !")
            break

# Add start to waypoints
rrt.Waypoints.insert(0, start)
print("Number of waypoints: ", rrt.numWaypoints)
print("Path Distance (m): ", rrt.path_distance)
print("Waypoints: ", rrt.Waypoints)

# plot the waypoints in red
for i in range(len(rrt.Waypoints) - 1):
    plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i + 1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i + 1][1]], 'ro',
             linestyle="--")
    plt.pause(0.10)
