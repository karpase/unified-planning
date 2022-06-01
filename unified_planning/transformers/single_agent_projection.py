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
"""This module defines the single agent projection transformer class."""

import unified_planning as up
from unified_planning.transformers.ab_transformer import ActionBasedTransformer
from unified_planning.model import Fluent, Problem, InstantaneousAction, DurativeAction, FNode, Action, Effect, Timing, Agent
from unified_planning.walkers.identitydag import IdentityDagWalker
from unified_planning.exceptions import UPExpressionDefinitionError, UPProblemDefinitionError
from typing import List, Dict, Union

class SingleAgentProjection(ActionBasedTransformer):
    '''Single agent projection class:
    this class requires a (multi agent) problem and an agent, and offers the capability
    to produce the single agent projection planning problem for the given agent.

    This is done by only including the actions of the given agent, changing waitfor preconditions to regular preconditions, and setting the goal to the agent's goal.'''
    def __init__(self, problem: Problem, agent: Agent, name: str = 'sap'):
        ActionBasedTransformer.__init__(self, problem, name)
                
        self._agent = agent
        #Represents the map from the new action to the old action
        self._new_to_old: Dict[Action, Action] = {}
        #represents a mapping from the action of the original problem to action of the new one.
        self._old_to_new: Dict[Action, List[Action]] = {}

    @property
    def agent(self) -> Agent:
        """Returns the agent."""
        return self._agent

    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that is a copy of the original problem
        but actions are modified and filtered.'''
        if self._new_problem is not None:
            return self._new_problem

        #NOTE that a different environment might be needed when multi-threading
        self._new_problem = Problem(f'{self._name}_{self._problem.name}', self._env)
        for o in self._problem.all_objects:
            self._new_problem.add_object(o)
        assert self._new_problem is not None

        name_action_map: Dict[str, Union[InstantaneousAction, DurativeAction]] = {}
        for action in self._problem.actions:
            if action.agent == self.agent:
                if isinstance(action, InstantaneousAction):
                    new_action = action.clone()
                    new_action.name = self.get_fresh_name(action.name)
                    new_action.clear_preconditions()
                    for p in action.preconditions:                    
                        new_action.add_precondition(p)
                    for p in action.preconditions_wait:                    
                        new_action.add_precondition(p)
                    self._old_to_new[action] = [new_action]
                    self._new_to_old[new_action] = action
                elif isinstance(action, DurativeAction):
                    raise NotImplementedError
                    # new_durative_action = action.clone()
                    # new_durative_action.name = self.get_fresh_name(action.name)
                    # new_durative_action.clear_conditions()
                    # for i, cl in action.conditions.items():
                    #     for c in cl:
                    #         nc = self._fluent_remover.remove_negative_fluents(c)
                    #         new_durative_action.add_condition(i, nc)
                    # for t, cel in new_durative_action.conditional_effects.items():
                    #     for ce in cel:
                    #         ce.set_condition(self._fluent_remover.remove_negative_fluents(ce.condition))
                    #self._old_to_new[action] = [new_durative_action]
                    #self._new_to_old[new_durative_action] = action
                else:
                    raise NotImplementedError

        for t, el in self._problem.timed_effects.items():
            for e in el:
                self._new_problem._add_effect_instance(t, e.clone())

        for i, gl in self._problem.timed_goals.items():
            for g in gl:                
                self._new_problem.add_timed_goal(i, gl)

        for g in self.agent.goals:            
            self._new_problem.add_goal(g)

        return self._new_problem

    def get_original_action(self, action: Action) -> Action:
        '''After the method get_rewritten_problem is called, this function maps
        the actions of the transformed problem into the actions of the original problem.'''
        return self._new_to_old[action]

    def get_transformed_actions(self, action: Action) -> List[Action]:
        '''After the method get_rewritten_problem is called, this function maps
        the actions of the original problem into the actions of the transformed problem.'''
        return self._old_to_new[action]
