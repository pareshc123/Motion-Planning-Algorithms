# RRT Star algorithm
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams

np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 10


# tree Node class
class TreeNode:

    def __init__(self, location_x, location_y):
        self.location_x = location_x  # x Location
        self.location_y = location_y  # y Location
        self.children = []  # children list
        self.parent = None  # parent node reference


# RRT Star Algorithm class
class RRTStarAlgorithm:

    def __init__(self, start_pos, goal_pos, num_iterations, grid_size, step_size):
        self.root_node = TreeNode(start_pos[0], start_pos[1])  # The RRT (root position)
        self.goal = TreeNode(goal_pos[0], goal_pos[1])  # goal position
        self.nearestNode = None  # nearest node
        self.iterations = min(num_iterations, 250)  # number of iterations to run
        self.grid = grid_size  # the map
        self.rho = step_size  # length of each branch
        self.path_distance = 0  # total path distance
        self.nearestDist = 10000  # distance to the nearest node (initialize with large)
        self.numWaypoints = 0  # number of waypoints
        self.Waypoints = []
        self.searchRadius = self.rho * 2  # the radius to search for finding neighbouring vertices
        self.neighbouringNodes = []  # neighbouring nodes
        self.goalArray = np.array([goal[0], goal[1]])  # goal as an array
        self.goalCosts = [10000]  # the costs to the goal (ignore first value)

    # add the node to the nearest node, and add goal if necessary
    def add_child(self, node):
        if node.location_x == self.goal.location_x:
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:

            self.nearestNode.children.append(node)
            node.parent = self.nearestNode

    @staticmethod
    # sample random point within grid limits
    def sample_new_point():
        x = random.randint(1, grid.shape[1])
        y = random.randint(1, grid.shape[0])
        new_point = np.array([x, y])
        return new_point

    # steer a distance stepSize from start location to end location
    def steer_to_new_point(self, location_start, location_end):
        offset = self.rho * self.unit_vector(location_start, location_end)
        new_point = np.array([location_start.location_x + offset[0], location_start.location_y + offset[1]])
        if new_point[0] >= grid.shape[1]:
            new_point[0] = grid.shape[1] - 1
        if new_point[1] >= grid.shape[0]:
            new_point[1] = grid.shape[0] - 1
        return new_point

    # check if obstacle lies between the start and end point of the edge
    def is_in_obstacle(self, location_start, location_end):
        u_hat = self.unit_vector(location_start, location_end)
        test_point = np.array([0.0, 0.0])
        for r in range(self.rho):
            test_point[0] = min(grid.shape[1] - 1, location_start.location_x + r * u_hat[0])
            test_point[1] = min(grid.shape[0] - 1, location_start.location_y + r * u_hat[1])
            if self.grid[round(test_point[1]), round(test_point[0])] == 1:
                return True
        return False

    @staticmethod
    # find the unit vector between 2 locations (DONE)
    def unit_vector(location_start, location_end):
        v = np.array([location_end[0] - location_start.location_x, location_end[1] - location_start.location_y])
        u_hat = v / np.linalg.norm(v)
        return u_hat

    # find the nearest node from a given (unconnected) point (Euclidean distance)
    def find_nearest(self, root_node, curr_point):

        if not root_node:
            return

        # find distance between root and point use distance method
        dist = self.distance(root_node, curr_point)

        # update the node
        if dist <= self.nearestDist:
            self.nearestNode = root_node
            self.nearestDist = dist

        # recursive call
        for child in root_node.children:
            self.find_nearest(child, curr_point)

    # find neighbouring nodes
    def find_neighbouring_nodes(self, root_node, curr_point):

        if not root_node:
            return

        # find distance between root and point use distance method
        dist = self.distance(root_node, curr_point)

        if dist <= self.searchRadius:
            self.neighbouringNodes.append(root_node)

        # recursive call
        for child in root_node.children:
            self.find_neighbouring_nodes(child, curr_point)

    @staticmethod
    # find euclidean distance between a node object and an XY point
    def distance(node1, curr_point):
        dist = math.sqrt((node1.location_x - curr_point[0]) ** 2 + (node1.location_y - curr_point[1]) ** 2)
        return dist

    # check if the goal is within step size (rho) distance from point, return true if so otherwise false
    def goal_found(self, curr_point):
        if self.distance(self.goal, curr_point) <= self.rho:
            return True
        return False

    # reset: set nearestNode to None and nearestDistance to 10000
    def reset_nearest_values(self):
        self.nearestNode = None
        self.nearestDist = 10000
        self.neighbouringNodes = []

    # trace the path from goal to start, since have to reset if called many times, do this iteratively
    def retrace_path(self):
        self.numWaypoints = 0
        self.Waypoints = []
        goal_cost = 0
        temp_goal = self.goal

        while temp_goal.location_x != self.root_node.location_x:
            self.numWaypoints += 1
            curr_point = np.array([temp_goal.location_x, temp_goal.location_y])
            self.Waypoints.insert(0, curr_point)
            goal_cost += self.distance(temp_goal, np.array([temp_goal.parent.location_x, temp_goal.parent.location_y]))
            temp_goal = temp_goal.parent  # set the node to it's parent

        self.goalCosts.append(goal_cost)

    # find unique path length from root of a node (cost)
    def find_path_distance(self, node):
        cost_from_root = 0
        current_node = node
        while current_node.location_x != self.root_node.location_x:
            cost_from_root += self.distance(current_node,
                                            np.array([current_node.parent.location_x, current_node.parent.location_y]))
            current_node = current_node.parent
        return cost_from_root


# load the grid, set start and goal <x, y> positions, number of iterations, step size
grid = np.load('cspace.npy')
start = np.array([300.0, 300.0])
goal = np.array([1400.0, 775.0])
numIterations = 700
stepSize = 75
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

fig = plt.figure("RRT Star Algorithm")
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
plt.xlabel('X-axis $(m)$')
plt.ylabel('Y-axis $(m)$')

# Begin
rrtStar = RRTStarAlgorithm(start, goal, numIterations, grid, stepSize)
plt.pause(2)

# RRT Star algorithm
# iterate
for i in range(rrtStar.iterations):

    # Reset the nearest values
    rrtStar.reset_nearest_values()
    print("Iteration: ", i)

    # sample a point
    point = rrtStar.sample_new_point()
    rrtStar.find_nearest(rrtStar.root_node, point)
    new = rrtStar.steer_to_new_point(rrtStar.nearestNode, point)  # steer to a point, return as 'new'

    # if not in obstacle
    if not rrtStar.is_in_obstacle(rrtStar.nearestNode, new):
        rrtStar.find_neighbouring_nodes(rrtStar.root_node, new)
        min_cost_node = rrtStar.nearestNode
        min_cost = rrtStar.find_path_distance(min_cost_node)
        min_cost = min_cost + rrtStar.distance(rrtStar.nearestNode, new)

        # connect along minimum cost path
        for branch in rrtStar.neighbouringNodes:
            branch_cost = rrtStar.find_path_distance(branch)
            branch_cost += rrtStar.distance(branch, new)

            if not rrtStar.is_in_obstacle(branch, new) and branch_cost < min_cost:
                min_cost_node = branch
                min_cost = branch_cost

        rrtStar.nearestNode = min_cost_node
        new_node = TreeNode(new[0], new[1])
        rrtStar.add_child(new_node)

        # plot for display
        plt.pause(0.01)
        plt.plot([rrtStar.nearestNode.location_x, new[0]], [rrtStar.nearestNode.location_y, new[1]], 'go',
                 linestyle="--")

        # rewire tree
        for branch in rrtStar.neighbouringNodes:
            branch_cost = min_cost
            branch_cost += rrtStar.distance(branch, new)

            if not rrtStar.is_in_obstacle(branch, new) and branch_cost < rrtStar.find_path_distance(branch):
                branch.parent = new_node

        # if goal found, and the projected cost is lower, then append to path let it sample more
        point = np.array([new_node.location_x, new_node.location_y])
        if rrtStar.goal_found(point):
            projectedCost = rrtStar.find_path_distance(new_node) + rrtStar.distance(rrtStar.goal, point)
            if projectedCost < rrtStar.goalCosts[-1]:
                rrtStar.add_child(rrtStar.goal)
                plt.plot([rrtStar.nearestNode.location_x, rrtStar.goalArray[0]],
                         [rrtStar.nearestNode.location_y, rrtStar.goalArray[1]], 'go', linestyle="--")
                # retrace and plot, this method finds waypoints and cost from root
                rrtStar.retrace_path()
                print("Goal Cost: ", rrtStar.goalCosts)
                plt.pause(0.25)
                rrtStar.Waypoints.insert(0, start)
                # plot the waypoints
                for waypoint in range(len(rrtStar.Waypoints) - 1):
                    plt.plot([rrtStar.Waypoints[waypoint][0], rrtStar.Waypoints[waypoint + 1][0]],
                             [rrtStar.Waypoints[waypoint][1], rrtStar.Waypoints[waypoint + 1][1]], 'ro', linestyle="--")
                    plt.pause(0.01)

print("Goal Costs: ", rrtStar.goalCosts)
