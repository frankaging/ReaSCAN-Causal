import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

from antra.antra import *
import json

def get_start_x_counter():
    def f(examples):
        result = []
        for example in examples:
            row =  int(example["situation"]["target_object"]["position"]["row"]) - int(example["situation"]["agent_position"]["row"])
            result.append(row)
        return result
    return f

def get_start_y_counter():
    def f(examples):
        result = []
        for example in examples:
            col = int(example["situation"]["target_object"]["position"]["column"]) - int(example["situation"]["agent_position"]["column"])
            result.append(col)
        return result
    return f

def get_start_o_counter():
    def f(examples):
        result = []
        for example in examples:
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


def get_counter_compgraph(
    max_decode_step,
    cache_results=False
):
    example = GraphNode.leaf(name=f"examples")
    fx = get_start_x_counter()
    fy = get_start_y_counter()
    fo = get_start_o_counter()
    fa = update_a()
    x_counter = GraphNode(example, name="x0", forward=fx)
    y_counter = GraphNode(example, name="y0", forward=fy)
    o_counter = GraphNode(example, name="o0", forward=fo)
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