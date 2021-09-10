import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

import json
import torch
import torch.nn as nn

class HighLevelModel(nn.Module):
    """
    HMM for gSCAN/ReaSCAN.
    """
    def __init__(
        self,
        grid_size=6
    ):
        super(HighLevelModel, self).__init__()
        self.grid_size = grid_size
        
        self.o_map = {
            "DOWN":0,
            "UP":1,
            "LEFT":2,
            "RIGHT":3,
            0:"DOWN",
            1:"UP",
            2:"LEFT",
            3:"RIGHT"
        }
        
        self.a_map = {
            "NULL":0,
            "walk":4,
            "turn left":5,
            "turn right":3,
            0:"NULL",
            4:"walk",
            5:"turn left",
            3:"turn right"
        }
        
        _rotate_left = {"RIGHT":"UP", "LEFT":"DOWN", "UP":"LEFT","DOWN":"RIGHT"}
        _rotate_right = {"UP":"RIGHT", "DOWN":"LEFT", "LEFT":"UP","RIGHT":"DOWN"}
        self.rotate_left = {}
        self.rotate_right = {}
        # also save the int form.
        for k, v in _rotate_left.items():
            self.rotate_left[self.o_map[k]] = self.o_map[v]
        for k, v in _rotate_right.items():
            self.rotate_right[self.o_map[k]] = self.o_map[v]

    def actions_list_to_sequence(self, actions_list, dim=","):
        return dim.join([self.a_map[a] for a in actions_list])
            
    def forward(
        self, 
        agent_positions_batch=None,
        target_positions_batch=None,
        hmm_states=None,
        hmm_actions=None,
        tag="_hmm_step_fxn"
    ):
        if tag == "situation_encode":
            agent_row = agent_positions_batch//self.grid_size
            agent_col = agent_positions_batch%self.grid_size
            target_row = target_positions_batch//self.grid_size
            target_col = target_positions_batch%self.grid_size
            row2target = target_row - agent_row
            col2target = target_col - agent_col
            orientation = torch.zeros_like(row2target).to(agent_positions_batch.device)
            init_state = torch.cat(
                [row2target, col2target, orientation], dim=-1
            )
            return init_state
        elif tag == "_hmm_step_fxn":
            
            new_hmm_states = hmm_states.clone()
            
            is_walking = hmm_actions == self.a_map["walk"]
            is_idling = hmm_actions == self.a_map["NULL"]
            is_lefting = hmm_actions == self.a_map["turn left"]
            is_righting = hmm_actions == self.a_map["turn right"]
            # ops on x
            is_facing_right = hmm_states[:,-1,None] == self.o_map["RIGHT"]
            is_facing_left = hmm_states[:,-1,None] == self.o_map["LEFT"]
            x_diff = (is_walking&is_facing_right).long() - (is_walking&is_facing_left).long()
            new_hmm_states[:,0,None] += x_diff
            
            # ops on y
            is_facing_up = hmm_states[:,-1,None] == self.o_map["UP"]
            is_facing_down = hmm_states[:,-1,None] == self.o_map["DOWN"]
            y_diff = (is_walking&is_facing_up).long() - (is_walking&is_facing_down).long()
            new_hmm_states[:,1,None] += y_diff
            
            # orientation lefting
            # for safety, let us copy current orientations and work on it not in-place.
            ori_rot_l = hmm_states[:,-1,None].clone()
            for k, v in self.rotate_left.items():
                ori_rot_l[hmm_states[:,-1,None]==k] = v
            ori_rot_l *= is_lefting.long()
            # orientation righting
            ori_rot_r = hmm_states[:,-1,None].clone()
            for k, v in self.rotate_right.items():
                ori_rot_r[hmm_states[:,-1,None]==k] = v
            ori_rot_r *= is_righting.long()
            # orientation staying
            ori_rot_null = hmm_states[:,-1,None].clone()
            ori_rot_null *= (is_walking|is_idling).long()
            new_hmm_states[:,-1,None] = ori_rot_l+ori_rot_r+ori_rot_null
            
            # actions
            new_hmm_actions = torch.zeros(hmm_states.size(0), 1).to(hmm_states.device)
            need_walk = (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]>0)&(new_hmm_states[:,2,None]==self.o_map["LEFT"])
            need_walk |= (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]<0)&(new_hmm_states[:,2,None]==self.o_map["RIGHT"])
            need_walk |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]>0)&(new_hmm_states[:,2,None]==self.o_map["DOWN"])
            need_walk |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]<0)&(new_hmm_states[:,2,None]==self.o_map["UP"])
            
            
            need_left = (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]>0)&((new_hmm_states[:,2,None]==self.o_map["RIGHT"])|(new_hmm_states[:,2,None]==self.o_map["UP"]))
            need_left |= (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]<0)&((new_hmm_states[:,2,None]==self.o_map["LEFT"])|(new_hmm_states[:,2,None]==self.o_map["DOWN"]))
            need_left |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]>0)&((new_hmm_states[:,2,None]==self.o_map["LEFT"])|(new_hmm_states[:,2,None]==self.o_map["UP"]))
            need_left |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]<0)&((new_hmm_states[:,2,None]==self.o_map["RIGHT"])|(new_hmm_states[:,2,None]==self.o_map["DOWN"]))
            
            need_right = (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]>0)&(new_hmm_states[:,2,None]==self.o_map["DOWN"])
            need_right |= (new_hmm_states[:,1,None]==0)&(new_hmm_states[:,0,None]<0)&(new_hmm_states[:,2,None]==self.o_map["UP"])
            need_right |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]>0)&(new_hmm_states[:,2,None]==self.o_map["RIGHT"])
            need_right |= (new_hmm_states[:,1,None]!=0)&(new_hmm_states[:,1,None]<0)&(new_hmm_states[:,2,None]==self.o_map["LEFT"])
            new_hmm_actions = need_walk.long()*self.a_map["walk"] + need_left.long()*self.a_map["turn left"] + need_right.long()*self.a_map["turn right"]
            
            return new_hmm_states, new_hmm_actions
        else:
            pass