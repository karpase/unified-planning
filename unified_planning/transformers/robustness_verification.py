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
"""This module defines the robustness verification transformer class."""

from unified_planning.transformers.transformer import Transformer
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import List, Dict


class RobustnessVerifier(Transformer):
    '''Robustness verification transformer class:
    this class requires a problem (with multiple agents, where a social law is already encoded)
    and outputs a classical planning problem whose solution encodes a counterexample to the robustness    
    '''
    def __init__(self, problem: Problem, name: str = 'slrob'):
        Transformer.__init__(self, problem, name)  
        self._new_problem = None      
        self._g_fluent_map = {}
        self._l_fluent_map = {}


    def get_global_version(self, fact):
        return self._new_problem._env.expression_manager.FluentExp(
            self._g_fluent_map[fact.fluent().name], 
            fact.args)

    def get_local_version(self, fact, agent):
        agent_tuple = (agent),
        return self._new_problem._env.expression_manager.FluentExp(
            self._l_fluent_map[fact.fluent().name], 
             agent_tuple + fact.args)


    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that implements the social law robustness verification compliation from Karpas, Shleyfman, Tennenholtz, ICAPS 2017 
        with the bugs fixed
        '''
        if self._new_problem is not None:
            return self._new_problem
        #NOTE that a different environment might be needed when multi-threading
        self._new_problem = Problem(f'{self._name}_{self._problem.name}')


        agent_type = UserType("agent")
        
        self._new_problem._add_user_type(agent_type)
        for type in self._problem.user_types:
            self._new_problem._add_user_type(type)

        self._new_problem.add_objects(self._problem.all_objects)


        failure = Fluent("failure")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", agent_type)])

        self._new_problem.add_fluent(failure, default_initial_value=False)
        self._new_problem.add_fluent(act, default_initial_value=False)
        self._new_problem.add_fluent(fin, default_initial_value=False)

        for f in self._problem.fluents:
            g_fluent = Fluent("g-" + f.name, f.type, f.signature)
            l_fluent = Fluent("l-" + f.name, f.type, [Parameter("agent", agent_type)] + f.signature)
            self._g_fluent_map[f.name] = g_fluent
            self._l_fluent_map[f.name] = l_fluent
            self._new_problem.add_fluent(g_fluent, default_initial_value=False)
            self._new_problem.add_fluent(l_fluent, default_initial_value=False)

        for agent in self._problem.agents:
            agent_object = unified_planning.model.Object(agent.name, agent_type)
            self._new_problem.add_object(agent_object)

            end_s = InstantaneousAction("end_s_" + agent.name)
            for goal in agent.goals:
                end_s.add_precondition(self.get_global_version(goal))
                end_s.add_precondition(self.get_local_version(goal, agent_object))
            end_s.add_effect(fin(agent_object), True)
            self._new_problem.add_action(end_s)


            for i, goal in enumerate(agent.goals):
                end_f = InstantaneousAction("end_f_" + agent.name + "_" + str(i))
                end_f.add_precondition(Not(self.get_global_version(goal)))
                for goal in agent.goals:
                    #end_s.add_precondition(self.get_global_version(goal))
                    end_f.add_precondition(self.get_local_version(goal, agent_object))
                end_f.add_effect(fin(agent_object), True)
                self._new_problem.add_action(end_f)

        
        # for action in self._new_problem.actions:
        #     if isinstance(action, InstantaneousAction):
        #         original_action = self._problem.action(action.name)
        #         assert isinstance(original_action, InstantaneousAction)
        #         action.name = self.get_fresh_name(action.name)
        #         action.clear_preconditions()
        #         for p in original_action.preconditions:
        #             action.add_precondition(self._expression_quantifier_remover.remove_quantifiers(p, self._problem))
        #         for e in action.effects:
        #             if e.is_conditional():
        #                 e.set_condition(self._expression_quantifier_remover.remove_quantifiers(e.condition, self._problem))
        #             e.set_value(self._expression_quantifier_remover.remove_quantifiers(e.value, self._problem))
        #         self._old_to_new[original_action] = [action]
        #         self._new_to_old[action] = original_action
        #     elif isinstance(action, DurativeAction):
        #         original_action = self._problem.action(action.name)
        #         assert isinstance(original_action, DurativeAction)
        #         action.name = self.get_fresh_name(action.name)
        #         action.clear_conditions()
        #         for i, cl in original_action.conditions.items():
        #             for c in cl:
        #                 action.add_condition(i, self._expression_quantifier_remover.remove_quantifiers(c, self._problem))
        #         for t, el in action.effects.items():
        #             for e in el:
        #                 if e.is_conditional():
        #                     e.set_condition(self._expression_quantifier_remover.remove_quantifiers(e.condition, self._problem))
        #                 e.set_value(self._expression_quantifier_remover.remove_quantifiers(e.value, self._problem))
        #         self._old_to_new[original_action] = [action]
        #         self._new_to_old[action] = original_action
        #     else:
        #         raise NotImplementedError
        # for t, el in self._new_problem.timed_effects.items():
        #     for e in el:
        #         if e.is_conditional():
        #             e.set_condition(self._expression_quantifier_remover.remove_quantifiers(e.condition, self._problem))
        #         e.set_value(self._expression_quantifier_remover.remove_quantifiers(e.value, self._problem))
        # for i, gl in self._problem.timed_goals.items():
        #     for g in gl:
        #         ng = self._expression_quantifier_remover.remove_quantifiers(g, self._problem)
        #         self._new_problem.add_timed_goal(i, ng)
        # for g in self._problem.goals:
        #     ng = self._expression_quantifier_remover.remove_quantifiers(g, self._problem)
        #     self._new_problem.add_goal(ng)
        # return self._new_problem
        return self._new_problem
