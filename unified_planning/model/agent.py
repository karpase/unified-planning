# Copyright 2022 AIPlan4EU project / Technion
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module defines the Agent base class and some of its extentions.
An Agent has a name, a goal, and a set of actions.
"""

import unified_planning as up
from unified_planning.environment import get_env, Environment
#from unified_planning.model.problem import Problem
from typing import List

class Agent:
    """This is the agent class."""
    def __init__(self, name: str = None, 
                    env: 'up.environment.Environment' = None,
                    goals: List['up.model.fnode.FNode'] = []):        
        self._name = name
        self._env = env
        self._goals = goals        

    @property
    def env(self) -> 'up.environment.Environment':
        '''Returns the problem environment.'''
        return self._env

    def __eq__(self, oth: object) -> bool:
        return isinstance(oth, Agent) and self._name == oth._name and self._goals == oth._goals

    def __hash__(self) -> int:
        return hash(self._name) + hash(self._goals)

    def clone(self):
        return Agent(self._name, self._env, self._goals)

    def __repr__(self) -> str:
        return "Agent [name=" + self._name + ", goal=" + str(self._goals) + "]"

    @property
    def name(self) -> str:
        """Returns the agent name."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        """Sets the agent name."""
        self._name = new_name

    @property
    def goals(self) -> List['up.model.fnode.FNode']:
        """Returns the agent goal."""
        return self._goals

    @name.setter
    def set_goals(self, new_goals: List['up.model.fnode.FNode']):
        """Sets the agent goal."""
        self._goals = new_goals

    def add_goal(self, goal: List['up.model.fnode.FNode']):
        """Adds a goal to the agent."""
        goal_exp, = self._env.expression_manager.auto_promote(goal)
        assert self._env.type_checker.get_type(goal_exp).is_bool_type()
        if goal_exp != self._env.expression_manager.TRUE():
            self._goals.append(goal_exp)

def get_agent_name_from_action(action: 'up.model.Action') -> str:
    """Guess the name of an agent performing given action
        If action has parameters, use first parameter value
        else, look for _ in action name (from grounding), and take first parameter (second  value)
        else, return null
    """
    if len(action.parameters) > 0:
        return action.parameters[0].name
    else:
        if "_" in action.name:
            return action.name.split("_")[1]
        else:
            return "null"

def get_agent_name_from_goal(goal: 'up.model.FNode') -> str:
    """Guess the name of an agent from a goal
        If goal has parameters, use first parameter value
        else, look for _ in goal fluent name (from grounding), and take first parameter (second  value)
        else, return null
    """
    if goal.is_fluent_exp():
        f = goal.fluent()
        if len(f.signature) > 0:
            return f.signature[0]
        else:
            if "_" in f.name:
                return f.name.split("_")[1]
            else:
                return "null"            
    else:
        return "null"
    

def defineAgentsByFirstArg(problem: 'up.model.Problem') -> List[Agent]:
    """Define agents by the first argument of all actions"""
    agents = {}
    
    for action in problem.actions:
        agent_name = get_agent_name_from_action(action)

        if agent_name not in agents.keys():
            agents[agent_name] = Agent(agent_name, problem.env, [], [])

        agents[agent_name].actions.append(action)

    for goal in problem.goals:
        agent_name = get_agent_name_from_goal(goal)

        if agent_name not in agents.keys():
            agents[agent_name] = Agent(agent_name, problem.env, [], [])

        agents[agent_name].goals.append(goal)

    return agents.values()