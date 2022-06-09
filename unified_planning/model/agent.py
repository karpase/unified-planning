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
from unified_planning.model.object import Object
#from unified_planning.model.problem import Problem
from typing import List

class Agent:
    """This is the basic agent class.
        It is implemented as a new object of type agent in the problem
    """
    def __init__(self, name: str = None, 
                    env: 'up.environment.Environment' = None,
                    goals: List['up.model.fnode.FNode'] = list(),
                    agent_type_name: str = "agent"):
        self._name = name
        self._env = env
        self._goals = goals
        self._agent_type_name = agent_type_name

    @property
    def env(self) -> 'up.environment.Environment':
        '''Returns the problem environment.'''
        return self._env

    def __eq__(self, oth: object) -> bool:
        return isinstance(oth, Agent) and self._name == oth._name and self._goals == oth._goals and self._agent_type_name == oth._agent_type_name

    def __hash__(self) -> int:
        return hash(self._name) + hash(self._goals)

    def clone(self):
        return Agent(self._name, self._env, self._goals, self._agent_type)

    def __repr__(self) -> str:
        return "Agent [name=" + self._name + ", goal=" + str(self._goals) + "]"

    @property
    def obj(self) -> 'up.model.object.Object':
        """Returns the agent object."""
        return Object(self.name, self.env.type_manager.UserType(self._agent_type_name))

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

    @goals.setter
    def set_goals(self, new_goals: List['up.model.fnode.FNode']):
        """Sets the agent goal."""
        self._goals = new_goals

    def add_goal(self, goal: List['up.model.fnode.FNode']):
        """Adds a goal to the agent."""
        goal_exp, = self._env.expression_manager.auto_promote(goal)
        assert self._env.type_checker.get_type(goal_exp).is_bool_type()
        if goal_exp != self._env.expression_manager.TRUE():
            self._goals.append(goal_exp)

    def add_obj_to_problem(self, problem: 'up.model.problem.Problem'):
        """adds the agent object to the given problem"""
        problem.add_object(self.obj)

class ExistingObjectAgent(Agent):
    """This is an agent class which is implemented as an already existing object.
        It is mainly used for lifted problems
    """
    def __init__(self, obj: 'up.model.object.Object', 
                    env: 'up.environment.Environment' = None,
                    goals: List['up.model.fnode.FNode'] = []):
        Agent.__init__(self, obj.name, env, goals)     
        self._obj = obj

    @property
    def obj(self) -> 'up.model.object.Object':
        """Returns the agent object."""
        return self._obj

    @obj.setter
    def set_obj(self, new_obj: 'up.model.object.Object'):
        """Sets the agent object."""
        self._obj = new_obj

    def __eq__(self, oth: object) -> bool:
        return isinstance(oth, ExistingObjectAgent) and self._obj == oth._obj

    def __hash__(self) -> int:
        return hash(self._obj) + hash(self._goals)

    def clone(self):
        return ExistingObjectAgent(self._obj, self._env, self._goals)

    def __repr__(self) -> str:
        return "ExistingObjectAgent [obj=" + str(self._obj) + ", goal=" + str(self._goals) + "]"

    @property
    def name(self) -> str:
        """Returns the agent name."""
        return self.obj.name

    def add_obj_to_problem(self, problem: 'up.model.problem.Problem'):
        """no need to add the agent object to the given problem, pass"""
        pass
  





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
        if len(goal.args) > 0:
            return str(goal.arg(0))
        else:
            if "_" in f.name:
                return f.name.split("_")[1]
            else:
                return "null"    
    elif goal.is_and_exp():
        return "null"
    else:
        return "null"
    

def defineAgentsByFirstArg(problem: 'up.model.Problem') -> List[Agent]:
    """Define agents by the first argument of all actions"""
    agents = {}
    
    for action in problem.actions:
        agent_name = "agent-" + get_agent_name_from_action(action)

        if agent_name not in agents.keys():
            agents[agent_name] = Agent(agent_name, problem.env, [])

        action.agent = agents[agent_name]

    #TODO: there must be a better way to do this using a walker
    atomic_goals = []
    for goal in problem.goals:
        if goal.is_and():
            atomic_goals += goal.args
        elif goal.is_fluent_exp():
            atomic_goals.append(goal)

    for goal in atomic_goals:
        agent_name = "agent-" + get_agent_name_from_goal(goal)

        if agent_name not in agents.keys():
            agents[agent_name] = Agent(agent_name, problem.env, [])

        agents[agent_name].add_goal(goal)

    for agent in agents.values():
        problem.add_agent(agent)