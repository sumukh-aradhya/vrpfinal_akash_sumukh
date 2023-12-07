#Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random as rand
from scipy.spatial import distance
import copy
import math
from deap import base, creator, tools
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

#Streamlit configuration (UI Framework)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("<h2 style='text-align: center;'>A Genetic Algorithm based</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Vehicle Routing Problem</h3>", unsafe_allow_html=True)
st.markdown("<p>Team Members: <i>Akash Janardhan Srinivas, Sumukh Naveen Aradhya</i></p>", unsafe_allow_html=True)
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

#Parameters setup for plots and visualizations
np.random.seed(3)
graph_size   = 15
graph_width  = 16
graph_height = 8

params = {'legend.fontsize': 'large','figure.figsize': (graph_width, graph_height),'axes.labelsize': graph_size,'axes.titlesize': graph_size,'xtick.labelsize': graph_size * 0.75,'ytick.labelsize': graph_size * 0.75,'axes.titlepad': 25}
plt.rcParams.update(params)
plt.rcParams.update(params)

#Configurations for VRP
total_cities = 27
total_clients = total_cities - 1
total_vehicles = 4
payload_per_vehicle = 7
center_box = (100, 200)

#Read and parse Christofides et al. 1979 (VRP-REP) dataset
df_from_csv = pd.read_csv('/Users/sumukharadhya/Desktop/vrp_cristo5.csv')
df_final = df_from_csv.drop(columns=['demand'])
list_of_lists = df_final.to_dict(orient='list')
df_final1 = pd.DataFrame(list_of_lists)
ndarray = df_final1.to_numpy()
coordinates_all_cities = ndarray

#Initialize names for cities and its coordinates, first coordinate represents Depot
names = [i for i in range(total_cities)]
names_clients = [i for i in range(1, total_cities)]
coordinates_dict = {name: coord for name,coord in zip(names, coordinates_all_cities)}
coordinates_dict_clients = {name: coord for name,coord in zip(names_clients, coordinates_all_cities[1:])}
plt.scatter(coordinates_all_cities[1:, 0], coordinates_all_cities[1:, 1], s=graph_size * 2, cmap='viridis')
plt.scatter(coordinates_all_cities[0, 0], coordinates_all_cities[0, 1], s=graph_size * 4, cmap='viridis')

#Calculate the distance matrix
distance_matrix = distance.cdist(coordinates_all_cities, coordinates_all_cities, 'euclidean')

#Initialize DEAP Library toolbox
deap_toolbox = base.Toolbox()

creator.create('Fitness_Function', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.Fitness_Function)

polar_angle = [0 for _ in range(total_cities)]

#Population initialization: Uses the sweep approach as elaborated in Billy E. Gillett, Leland R. Miller, (1974) A Heuristic Algorithm for the Vehicle-Dispatch Problem
def init_population(_cities_names):
    for k,v in coordinates_dict.items():
        for i in _cities_names:
            if k==i:
                polar_angle[i]=math.atan2(v[0], v[1])

    schedule_of_cities = copy.deepcopy(_cities_names)
    vehicle_list = list(np.random.randint(total_vehicles, size=(len(schedule_of_cities))))
    paired_list = zip(_cities_names, polar_angle)
    sorted_paired_list = sorted(paired_list, key=lambda x: x[1]) #Sort the cities in increasing order of their polar angle
    schedule_of_cities = [element[0] for element in sorted_paired_list]
    chromosome = [schedule_of_cities, vehicle_list]

    return chromosome

#Evaluate each chromozome that is considered
def chromosome_evaluate(_dist_matrix, _chromo):
    routes = [[] for _ in range(total_vehicles)]
    for schedule, vehicle in zip(_chromo[0], _chromo[1]):
        routes[vehicle].append(schedule)

    distance = 0
    for route in routes:
        distance += calculate_total_cost_route(_dist_matrix, route)

    return distance,


#Obtain route for a particular chromozome
def route_get(_chromosome):
    routes = [[] for _ in range(total_vehicles)]
    for schedule, vehicle in zip(_chromosome[0], _chromosome[1]):
        routes[vehicle].append(schedule)
    return routes


#Method to calculate the total distane that would be travlled for a particular route that is considered
def calculate_total_cost_route(distance_matrix, route):
    if not route:
        return 0
    distance = distance_matrix[route[-1], 0] + distance_matrix[0, route[0]]

    for p in range(len(route) - 1):
        i = route[p]
        j = route[p + 1]
        distance += distance_matrix[i][j]

    return distance


#Crossover method definition
def crossover(chromosome1, chromosome2):
    split = get_chromosome_split()
    partial_crossover(chromosome1[0], chromosome2[0], split)

    split1 = get_chromosome_split()
    split2 = get_chromosome_split(split1[2])

    gene_swap(chromosome1[1], chromosome2[1], split1, split2)


#Helper method - partially mapped crossover for cities, used by genetic algo crossover method
def partial_crossover(chromosome1, chromosome2, split):

    size = len(chromosome1)
    parent1, parent2 = [0] * size, [0] * size

    for i in range(size):
        parent1[chromosome1[i] - 1] = i
        parent2[chromosome2[i] - 1] = i

    for i in range(split[0], split[1]):

        temp1 = chromosome1[i] - 1
        temp2 = chromosome2[i] - 1

        chromosome1[i], chromosome1[parent1[temp2]] = temp2 + 1, temp1 + 1
        chromosome2[i], chromosome2[parent2[temp1]] = temp1 + 1, temp2 + 1

        parent1[temp1], parent1[temp2] = parent1[temp2], parent1[temp1]
        parent2[temp1], parent2[temp2] = parent2[temp2], parent2[temp1]


#Helper methods - 2 pointer crossover method for ehicles, used by genetic algo crossover method
def get_chromosome_split(split_range=None, mutation=False):

    if mutation:
        randrange = total_clients
    else:
        randrange = total_clients + 1

    if split_range is None:
        split1 = rand.randrange(randrange)
        split2 = rand.randrange(randrange)
        if split1 > split2:
            tmp = split2
            split2 = split1
            split1 = tmp
        split_range = split2 - split1
    else:

        split1 = rand.randrange(total_clients + 1 - split_range)
        split2 = split1 + split_range
    return split1, split2, split_range


def gene_swap(chromosome1, chromosome2, split1, split2):
    tmp = chromosome1[split1[0]:split1[1]]
    chromosome1[split1[0]:split1[1]] = chromosome2[split2[0]:split2[1]]
    chromosome2[split2[0]:split2[1]] = tmp

#Mutation method definition - uses one of two methods with 50% probability, either swaps the genes or shuffles the genes
def mutate(_chromo):
    if np.random.rand() < 0.5:
        gene_swap_mutate(_chromo)
    else:
        gene_shuffle(_chromo)


#Helper method for mutation to swap genes
def gene_swap_mutate(chromosome):
    split = get_chromosome_split(mutation=True)

    if np.random.rand() < 0.5:
        tmp = chromosome[0][split[0]]
        chromosome[0][split[0]] = chromosome[0][split[1]]
        chromosome[0][split[1]] = tmp
    else:
        tmp = chromosome[1][split[0]]
        chromosome[1][split[0]] = chromosome[1][split[1]]
        chromosome[1][split[1]] = tmp


#Helper method for mutation to shuffle genes
def gene_shuffle(chromosome):
    split = get_chromosome_split(mutation=True)

    if np.random.rand() < 0.5:
        tmp = chromosome[0][split[0]:split[1]]
        np.random.shuffle(tmp)
        chromosome[0][split[0]:split[1]] = tmp
    else:
        tmp = chromosome[1][split[0]:split[1]]
        np.random.shuffle(tmp)
        chromosome[1][split[0]:split[1]] = tmp


def feasibility_check(chromosome):
    extra_payload = [payload_per_vehicle - chromosome[1].count(i) for i in range(total_vehicles)]
    Vehicle_id = [i for i in range(total_vehicles)]

    while any(_p < 0 for _p in extra_payload):
        vehicle_id = next(i for i,j in enumerate(extra_payload) if j < 0)
        available_vehicles = [i for i,j in enumerate(extra_payload) if j > 0]

        if len(available_vehicles) == 0:
            raise('INFEASIBLE SOLUTION: No available vehicle to accept excess cargo. Increase the number of vehicles or the vehcile payload')

        index = [i for i, x in enumerate(chromosome[1]) if x == vehicle_id]
        next_vehicle = rand.choice(available_vehicles)
        index_to_move = rand.choice(index)
        chromosome[1][index_to_move] = next_vehicle
        extra_payload[vehicle_id] += 1
        extra_payload[next_vehicle] -= 1


#Initialize all gentic algo methods into the DEAP framework
deap_toolbox.register('indeces', init_population, names_clients)
deap_toolbox.register('individual', tools.initIterate, creator.Individual, deap_toolbox.indeces)
deap_toolbox.register('population', tools.initRepeat, list, deap_toolbox.individual)
deap_toolbox.register('evaluate', chromosome_evaluate, distance_matrix)
deap_toolbox.register('select', tools.selTournament)
deap_toolbox.register('mate', crossover)
deap_toolbox.register('mutate', mutate)
deap_toolbox.register('feasibility', feasibility_check)
population_number = 200
generations_number = 1000
crossover_probability = .4
mutation_probability = .6

population = deap_toolbox.population(n=population_number)

fitness_set = list(deap_toolbox.map(deap_toolbox.evaluate, population))
for ind, fit in zip(population, fitness_set):
    ind.fitness.values = fit

best_fit_list = []
best_sol_list = []

#Accelerating the convergence of GA by using neighborhood search algorithm - using 2 optimal search as per Barrie M. Baker, M.A. Ayechew, ‘A genetic algorithm for the vehicle routing problem, Computers & Operations Research’
def two_opt_swap(route, i, k):
    return route[:i] + route[i:k+1][::-1] + route[k+1:]

def apply_two_opt_to_routes(routes, dist_matrix, max_iterations=50):
    """Apply 2-opt algorithm with limited iterations to improve each route."""
    for route_index, route in enumerate(routes):
        iteration = 0
        while iteration < max_iterations:
            best_distance = calculate_total_cost_route(dist_matrix, route)
            start_improvement = False

            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = two_opt_swap(route, i, j)
                    new_distance = calculate_total_cost_route(dist_matrix, new_route)
                    if new_distance < best_distance:
                        routes[route_index] = new_route
                        best_distance = new_distance
                        start_improvement = True
                        # Exit inner loop early as improvement is found
                        break

                if start_improvement:
                    # Exit outer loop early and go to the next iteration
                    break

            if not start_improvement:
                # No improvement found, no need for further iterations
                break

            iteration += 1

    return routes


st.markdown("<hr></hr>", unsafe_allow_html=True)

#Initialization for OR-Tools
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = distance_matrix.tolist()
    data["num_vehicles"] = 4
    data["depot"] = 0
    return data

st.markdown("<h2 style='text-align: center;'>Running ORTools Optimal Algo...</h2>", unsafe_allow_html=True)
st.markdown("<h3><b>Total Distance for this route: </b></h3>", unsafe_allow_html=True)

scale_factor = 10000  # Scale the distance to convert it to an integer
data = create_data_model()

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(
    len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
)

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)

# Create and register a transit callback.
def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    distance = data["distance_matrix"][from_node][to_node]
    return int(distance * scale_factor)
transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define cost of each arc
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Distance constraint
dimension_name = "Distance"
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    300000000,  # vehicle maximum travel distance
    True,  # start cumul to zero
    dimension_name,
)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

# Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if solution:
    max_route_distance = 0
    total_distance_of_routes = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance/scale_factor}\n"
        total_distance_of_routes += route_distance
        st.write(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    st.write(f"Maximum of the route distances: {max_route_distance/scale_factor}")
    st.markdown("<h3><b>Total Distance for this route: </b></h3>", unsafe_allow_html=True)
    st.text(total_distance_of_routes/scale_factor)
    or_route_distance = total_distance_of_routes/scale_factor
else:
    st.write("No solution found !")

st.markdown("<hr></hr>", unsafe_allow_html=True)

#Call all methods onto the UI for processing
def notebook_cell_1():
    best_fitting = np.Inf
    for generation in range(0, generations_number):
        if (generation % 50 == 0):
            st.write(f'Generation: {generation:4} | Fitness: {best_fitting:.2f}' )
        child = deap_toolbox.select(population, len(population), tournsize=3)
        child = list(map(deap_toolbox.clone, child))
        for c1, c2 in zip(child[0::2], child[1::2]):
            if np.random.random() < crossover_probability:
                deap_toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values
        for chromosome in child:
            if np.random.random() < mutation_probability:
                deap_toolbox.mutate(chromosome)
                del chromosome.fitness.values
        for chromosome in child:
            deap_toolbox.feasibility(chromosome)
        invalid_index = [index for index in child if not index.fitness.valid]
        fitness_set = map(deap_toolbox.evaluate, invalid_index)
        for index, fit in zip(invalid_index, fitness_set):
            index.fitness.values = fit
        population[:] = child
        current_best_solution = tools.selBest(population, 1)[0]
        current_best_fittness = current_best_solution.fitness.values[0]
        if current_best_fittness < best_fitting:
            best_sol = current_best_solution
            best_fitting = current_best_fittness
        best_fit_list.append(best_fitting)
        best_sol_list.append(best_sol)
    plt.plot(best_fit_list)
    st.pyplot(plt.show())

    best_routes = route_get(best_sol)
    temp_best_routes = copy.deepcopy(best_routes)
    optimized_routes = apply_two_opt_to_routes(temp_best_routes, distance_matrix)

    def calculate_total_distance(routes, distance_matrix):
        total_distance = 0
        for route in routes:
            if not route:
                continue
            total_distance += distance_matrix[0][route[0]]
            for i in range(len(route)-1):
                total_distance += distance_matrix[route[i]][route[i + 1]]
            total_distance += distance_matrix[route[-1]][0]
        return total_distance
    st.markdown("<hr></hr>", unsafe_allow_html=True)
    # Pure GA functionality calls - plots on UI
    st.markdown("<h2 style='text-align: center;'>Running Pure Genetic Algo...</h2>", unsafe_allow_html=True)
    st.write('Route iteration 1:',str(best_routes))
    st.markdown("<h3><b>Total Distance for this route: </b></h3>", unsafe_allow_html=True)
    pg_route_distance = calculate_total_distance(best_routes, distance_matrix.tolist())
    st.write(str(pg_route_distance))
    plt.scatter(coordinates_all_cities[1:, 0], coordinates_all_cities[1:, 1], s=graph_size * 2, cmap='viridis')
    plt.scatter(coordinates_all_cities[0, 0], coordinates_all_cities[0, 1], s=graph_size * 4, cmap='viridis')
    for i, txt in enumerate(names):
        plt.annotate(txt, (coordinates_all_cities[i, 0] + 1, coordinates_all_cities[i, 1] + 1))

    for r in best_routes:
        route = [0] + r + [0]
        for p in range(len(route) - 1):
            i = route[p]
            j = route[p + 1]
            colour = 'black'
            plt.arrow(coordinates_all_cities[i][0],
                      coordinates_all_cities[i][1],
                      coordinates_all_cities[j][0] - coordinates_all_cities[i][0],
                      coordinates_all_cities[j][1] - coordinates_all_cities[i][1],
                      color=colour)
    st.pyplot(plt.show())
    st.markdown("<hr></hr>", unsafe_allow_html=True)

    #Hyrbid enhanced GA functionality calls - plots on UI
    st.markdown("<h2 style='text-align: center;'>Running Optimized Genetic Algo...</h2>", unsafe_allow_html=True)
    st.write('Route iteration 2:', str(optimized_routes))
    st.markdown("<h3><b>Total Distance for this route: </b></h3>", unsafe_allow_html=True)
    og_route_distance = calculate_total_distance(optimized_routes, distance_matrix.tolist())
    st.write(str(og_route_distance))
    plt.scatter(coordinates_all_cities[1:, 0], coordinates_all_cities[1:, 1], s=graph_size * 2, cmap='viridis')
    plt.scatter(coordinates_all_cities[0, 0], coordinates_all_cities[0, 1], s=graph_size * 4, cmap='viridis')
    for i, txt in enumerate(names):
        plt.annotate(txt, (coordinates_all_cities[i, 0] + 1, coordinates_all_cities[i, 1] + 1))

    for r in optimized_routes:
        route = [0] + r + [0]
        for p in range(len(route) - 1):
            i = route[p]
            j = route[p + 1]
            colour = 'black'
            plt.arrow(coordinates_all_cities[i][0],
                      coordinates_all_cities[i][1],
                      coordinates_all_cities[j][0] - coordinates_all_cities[i][0],
                      coordinates_all_cities[j][1] - coordinates_all_cities[i][1],
                      color=colour)
    st.pyplot(plt.show())
    st.markdown("<hr></hr>", unsafe_allow_html=True)
    plt.style.use('dark_background')
    st.markdown("<h2 style='text-align: center;'>Plotting Results Based on Above Output...</h2>", unsafe_allow_html=True)
    categories = ['OR-Tools', 'Pure Genetic Algo', 'Optimized Genetic Algo']
    values = [or_route_distance, pg_route_distance, og_route_distance]
    bar_width = 0.3
    colors = ['#a7c957', '#386641', '#6a994e']

    plt.bar(categories, values, width=bar_width, color=colors)

    plt.title('Route distance for VRP')
    plt.xlabel('Algorithms')
    plt.ylabel('Total Distance')

    st.pyplot(plt.show())
    st.markdown("<hr></hr>", unsafe_allow_html=True)

#Define HTML button to run all GA models
if st.button('Run GA models'):
    notebook_cell_1()