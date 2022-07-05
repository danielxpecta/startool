from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import plotly.express as px
from datetime import date, datetime, time, timedelta
import numpy as np
from sklearn.neighbors import DistanceMetric
token = 'pk.eyJ1IjoidmFsZW50aW5hbGF2ZXJkZSIsImEiOiJja3lkODV6dnAwMzNtMnVxaGx1ZjNobjUzIn0.q98o5a_hvqz2O4RI3DNmTw'# MAPBOX

def create_google_data(drive_list):
    dist = DistanceMetric.get_metric('haversine')
    df = drive_list[['id','latitude','longitude']].set_index('id')
    probability = drive_list[['id','Probability']].set_index('id')
    matriz_distancia = pd.DataFrame(dist.pairwise(df)*6373,  columns=df.index, index=df.index)
    matriz_tiempo = (matriz_distancia/50).round(2)

    for i in matriz_distancia.index:
        try:
            matriz_distancia[str(i)]=matriz_distancia[str(i)]*(1-float(probability.loc[i,'Probability']))
        except:
            pass
    data= {}
    data['time_matrix'] = matriz_tiempo.astype(int).values
    data['distancia'] =  matriz_distancia.astype(int).values
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def solve_route(data):
    manager = pywrapcp.RoutingIndexManager(len(data['distancia']),data['num_vehicles'],data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    # Se asigna como pesos del grafo la matriz de distancia
    def weight_callback(from_index, to_index):
        """Returns the weight between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distancia'][from_node][to_node]



    transit_callback_index = routing.RegisterTransitCallback(weight_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Se añade la matriz de tiempo al modelo
    def time_callback(from_index, to_index):
        """Returns the time between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node] 

    transit_callback_index_0 = routing.RegisterTransitCallback(time_callback)

    #Se añade la restricción de tiempo al modelo para que la ruta no pase los 420 minutos
    dim = 'Tiempo'
    routing.AddDimension(transit_callback_index_0,
            0,  # no slack
            420,  # vehicle maximum travel time
            True,  # start cumul to zero
            dim)
    time_dimension = routing.GetDimensionOrDie(dim)
    time_dimension.SetGlobalSpanCostCoefficient(100)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    #Se resuelve la ruta
    solution = routing.SolveWithParameters(search_parameters)
    return solution,manager,routing

def print_solution(data, manager, routing, solution,clientes):
    """Prints solution on console."""
    max_route_distance = 0
    df=pd.DataFrame()

    #Los vehiculos en este caso son tomados como los dias necesarios para recorrer todas las cuentas
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        clientes_=0
        #Se crea dataframe con la información de rutas de cada dia
        data=pd.DataFrame(columns=['Id','Practice Name','Buy Probability','Last Visit Outcome','City','Zip','Phone','Last Buy Date','Last Visit Date'])
        while not routing.IsEnd(index):
            clientes_+=1
            
            df2 = {
                'Id':clientes.loc[manager.IndexToNode(index),'id'],
                'Practice Name': clientes.loc[manager.IndexToNode(index),'Practice Name'],
                'Buy Probability': clientes.loc[manager.IndexToNode(index),'Buy Probability'],
                'Last Visit Outcome':clientes.loc[manager.IndexToNode(index),'last_visit_outcome'],
                'City': clientes.loc[manager.IndexToNode(index),'city'],
                'Zip': clientes.loc[manager.IndexToNode(index),'Zip'],
                'Phone': clientes.loc[manager.IndexToNode(index),'phone'],
                'Last Buy Date':clientes.loc[manager.IndexToNode(index),'Last Buy Date'],
                'Last Visit Date':clientes.loc[manager.IndexToNode(index),'Last Visit Date']
                }
            data = data.append(df2, ignore_index = True)
            index = solution.Value(routing.NextVar(index))
        df=df.append(data,ignore_index=True)
    return df