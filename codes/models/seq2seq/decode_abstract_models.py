import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

from antra.antra import *
import json
import torch

def index2corrdinates(index:int, 
                      grid_size=6):
    row = index//grid_size
    col = index%grid_size
    return (row, col)

def get_start_x_counter():
    def f(agent_positions_batch, target_positions_batch):
        result = []
        batch_szie = agent_positions_batch.shape[0]
        for i in range(0, batch_szie):
            row = index2corrdinates(target_positions_batch[i])[0] - \
                    index2corrdinates(agent_positions_batch[i])[0]
            result.append(row)
        return result
    return f

def get_start_y_counter():
    def f(agent_positions_batch, target_positions_batch):
        result = []
        batch_szie = agent_positions_batch.shape[0]
        for i in range(0, batch_szie):
            col = index2corrdinates(target_positions_batch[i])[1] - \
                    index2corrdinates(agent_positions_batch[i])[1]
            result.append(col)
        return result
    return f

def get_start_o_counter():
    def f(agent_positions_batch, target_positions_batch):
        result = []
        batch_szie = agent_positions_batch.shape[0]
        for i in range(0, batch_szie):
            result.append("DOWN")
        return result
    return f

def update_x():
    def f(xs,az,os):
        results = []
        for x,a,o in zip(xs,az,os):
            if o =="RIGHT" and a == "walk":
                results.append(x + 1)
            elif o =="LEFT" and a == "walk":
                results.append(x - 1)
            else:
                results.append(x)
        return results
    return f

def update_y():
    def f(ys,az,os):
        results = []
        for y,a,o in zip(ys,az,os):
            if o =="UP" and a == "walk":
                results.append(y + 1)
            elif o =="DOWN" and a == "walk":
                results.append(y - 1)
            else:
                results.append(y)
        return results
    return f

def update_o():
    def f(az,os):
        results = []
        for a,o in zip(az,os):
            rotate_left = {"RIGHT":"UP", "LEFT":"DOWN", "UP":"LEFT","DOWN":"RIGHT"}
            rotate_right = {"UP":"RIGHT", "DOWN":"LEFT", "LEFT":"UP","RIGHT":"DOWN"}
            if a == "walk" or a == "NULL":
                results.append(o)
            elif a == "turn left":
                results.append(rotate_left[o])
            elif a == "turn right":
                results.append(rotate_right[o])
        return results
    return f

def update_a():
    def f(xs, ys, os):
        results = []
        for x, y, o in zip(xs,ys,os):
            if y == 0 and x ==0:
                results.append("NULL")
                continue
            if y == 0:
                if x > 0:
                    if o == "LEFT":
                        results.append("walk")
                    elif o in ["RIGHT", "UP"]:
                        results.append("turn left")
                    elif o == "DOWN":
                        results.append("turn right")
                if x < 0:
                    if o == "RIGHT":
                        results.append("walk")
                    elif o  == "UP":
                        results.append("turn right")
                    elif o in ["LEFT", "DOWN"]:
                        results.append("turn left")
            else:
                if y > 0:
                    if o == "DOWN":
                        results.append("walk")
                    elif o in ["UP", "LEFT"]:
                        results.append("turn left")
                    elif o == "RIGHT":
                        results.append("turn right")
                if y < 0:
                    if o == "UP":
                        results.append("walk")
                    elif o in ["DOWN", "RIGHT"]:
                        results.append("turn left")
                    elif o == "LEFT":
                        results.append("turn right")
        return results
    return f

def get_start_state():
    def f(agent_positions_batch, target_positions_batch):
        results = []
        batch_size = len(agent_positions_batch)
        for i in range(0, batch_size):
            x = index2corrdinates(target_positions_batch[i][0])[0] - \
                    index2corrdinates(agent_positions_batch[i][0])[0]
            y = index2corrdinates(target_positions_batch[i][0])[1] - \
                    index2corrdinates(agent_positions_batch[i][0])[1]
            o = 0 # default DOWN
            a = 0 # default NULL
            results.append([x, y, o, a])
        return torch.tensor(results)
    return f

def update_state():
    def f(prev_s):
        
        o_map = {
            "DOWN":0,
            "UP":1,
            "LEFT":2,
            "RIGHT":3,
            0:"DOWN",
            1:"UP",
            2:"LEFT",
            3:"RIGHT"
        }
        
        a_map = {
            "NULL":0,
            "walk":4,
            "turn left":5,
            "turn right":3,
            0:"NULL",
            4:"walk",
            5:"turn left",
            3:"turn right"
        }
        
        results = []
        batch_size = len(prev_s)
        for i in range(0, batch_size):
            prev_x = int(prev_s[i][0])
            prev_y = int(prev_s[i][1])
            prev_o = int(prev_s[i][2])
            prev_a = int(prev_s[i][3])

            # update x
            if o_map[prev_o] =="RIGHT" and a_map[prev_a] == "walk":
                x = prev_x + 1
            elif o_map[prev_o] =="LEFT" and a_map[prev_a] == "walk":
                x = prev_x - 1
            else:
                x = prev_x
            
            # update y
            if o_map[prev_o] =="UP" and a_map[prev_a] == "walk":
                y = prev_y + 1
            elif o_map[prev_o] =="DOWN" and a_map[prev_a] == "walk":
                y = prev_y - 1
            else:
                y = prev_y
            
            # update o
            rotate_left = {"RIGHT":"UP", "LEFT":"DOWN", "UP":"LEFT","DOWN":"RIGHT"}
            rotate_right = {"UP":"RIGHT", "DOWN":"LEFT", "LEFT":"UP","RIGHT":"DOWN"}
            if a_map[prev_a] == "walk" or a_map[prev_a] == "NULL":
                o = prev_o
            elif a_map[prev_a] == "turn left":
                o = o_map[rotate_left[o_map[prev_o]]]
            elif a_map[prev_a] == "turn right":
                o = o_map[rotate_right[o_map[prev_o]]]
            
            # update a
            if y == 0 and x == 0:
                a = a_map["NULL"]
            if y == 0:
                if x > 0:
                    if o_map[o] == "LEFT":
                        a = a_map["walk"]
                    elif o_map[o] in ["RIGHT", "UP"]:
                        a = a_map["turn left"]
                    elif o_map[o] == "DOWN":
                        a = a_map["turn right"]
                if x < 0:
                    if o_map[o] == "RIGHT":
                        a = a_map["walk"]
                    elif o_map[o]  == "UP":
                        a = a_map["turn right"]
                    elif o_map[o] in ["LEFT", "DOWN"]:
                        a = a_map["turn left"]
            else:
                if y > 0:
                    if o_map[o] == "DOWN":
                        a = a_map["walk"]
                    elif o_map[o] in ["UP", "LEFT"]:
                        a = a_map["turn left"]
                    elif o_map[o] == "RIGHT":
                        a = a_map["turn right"]
                if y < 0:
                    if o_map[o] == "UP":
                        a = a_map["walk"]
                    elif o_map[o] in ["DOWN", "RIGHT"]:
                        a = a_map["turn left"]
                    elif o_map[o] == "LEFT":
                        a = a_map["turn right"]
            results.append([x, y, o, a])
        return torch.tensor(results)
    return f

def get_counter_compgraph(
    max_decode_step,
    cache_results=False,
    tag="state"
):
    if tag == "state":
        agent_positions_batch = GraphNode.leaf(name=f"agent_positions_batch")
        target_positions_batch = GraphNode.leaf(name=f"target_positions_batch")
        fs = get_start_state()
        _s = GraphNode(agent_positions_batch, target_positions_batch, name="s0", forward=fs)
        for i in range(1, max_decode_step+1):
            fs = update_state()
            new_s = GraphNode(_s, name="s"+str(i), forward=fs)
            _s = new_s
        return ComputationGraph(_s)
    else:
        agent_positions_batch = GraphNode.leaf(name=f"agent_positions_batch")
        target_positions_batch = GraphNode.leaf(name=f"target_positions_batch")
        fx = get_start_x_counter()
        fy = get_start_y_counter()
        fo = get_start_o_counter()
        fa = update_a()
        x_counter = GraphNode(agent_positions_batch, target_positions_batch, name="x0", forward=fx)
        y_counter = GraphNode(agent_positions_batch, target_positions_batch, name="y0", forward=fy)
        o_counter = GraphNode(agent_positions_batch, target_positions_batch, name="o0", forward=fo)
        action = GraphNode(x_counter,y_counter,o_counter, name="a0", forward=fa)
        for i in range(1, max_decode_step+1):
            fx = update_x()
            fy = update_y()
            fa = update_a()
            fo = update_o()
            new_x_counter = GraphNode(x_counter, action, o_counter, name="x"+str(i), forward=fx)
            new_y_counter = GraphNode(y_counter,action,o_counter, name="y"+str(i), forward=fy)
            new_o_counter = GraphNode(action,o_counter, name="o"+str(i), forward=fo)
            new_action = GraphNode(new_x_counter, new_y_counter,new_o_counter, name="a"+str(i), forward=fa)
            x_counter = new_x_counter
            y_counter = new_y_counter
            o_counter = new_o_counter
            action = new_action
        return ComputationGraph(action)