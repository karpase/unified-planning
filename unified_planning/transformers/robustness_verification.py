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
        self._w_fluent_map = {}        
        self.act_pred = None

    def get_global_version(self, fact):
        """get the global copy of given fact
        """
        #TODO: there must be a cleaner way to do this...
        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        gfact = self._new_problem._env.expression_manager.FluentExp(
            self._g_fluent_map[fact.fluent().name], 
            fact.args)
        if negate:
            return Not(gfact)
        else:
            return gfact

    def get_local_version(self, fact, agent):
        """get the local copy of agent <agent> of given fact
        """
        agent_tuple = (agent),

        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        lfact = self._new_problem._env.expression_manager.FluentExp(
            self._l_fluent_map[fact.fluent().name], 
            agent_tuple + fact.args)
        if negate:
            return Not(lfact)
        else:
            return lfact

    def get_waiting_version(self, fact):
        """get the waiting copy of given fact
        """
        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        wfact = self._new_problem._env.expression_manager.FluentExp(
            self._w_fluent_map[fact.fluent().name], 
            fact.args)
        if negate:
            return Not(wfact)
        else:
            return wfact            


    def create_action_copy(self, action, suffix):
        """Create a new copy of an action, with name action_name_suffix, and duplicates the local preconditions/effects
        """        
        if len(action.parameters) == 0:
            new_action = InstantaneousAction(action.name + suffix)
        else:
            d = {}
            for p in action.parameters:
                d[p.name] = p.type
            new_action = InstantaneousAction(action.name + suffix, _parameters=d)

        if action.agent.name == "a":
            agent_object = new_action.parameters[0]
        else:
            agent_object = action.agent.obj

        new_action.add_precondition(self.act_pred)
        
        #TODO: can probably do this better with a substitution walker
        for fact in action.preconditions + action.preconditions_wait:
            if fact.is_and():
                for f in fact.args:
                    new_action.add_precondition(self.get_local_version(f, agent_object))
            else:
                new_action.add_precondition(self.get_local_version(fact, agent_object))
        for effect in action.effects:                
            new_action.add_effect(self.get_local_version(effect.fluent, agent_object), effect.value)
        return new_action

    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that implements the social law robustness verification compliation from Karpas, Shleyfman, Tennenholtz, ICAPS 2017 
        with the bugs fixed
        '''
        if self._new_problem is not None:
            return self._new_problem
        self._new_problem = Problem(f'{self._name}_{self._problem.name}')


        
        agent_type = self._problem.agents[0].obj.type
        self._new_problem._add_user_type(agent_type)

        for type in self._problem.user_types:
            self._new_problem._add_user_type(type)

        self._new_problem._add_user_type(UserType("agent"))
        self._new_problem.user_type("car")._father = UserType("agent")


        

        self._new_problem.add_objects(self._problem.all_objects)


        failure = Fluent("failure")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", agent_type)])

        self.act_pred = act
        

        self._new_problem.add_fluent(failure, default_initial_value=False)
        self._new_problem.add_fluent(act, default_initial_value=True)
        self._new_problem.add_fluent(fin, default_initial_value=False)
        self._new_problem.add_fluent(waiting, default_initial_value=False)

        for f in self._problem.fluents:
            g_fluent = Fluent("g-" + f.name, f.type, f.signature)
            l_fluent = Fluent("l-" + f.name, f.type, [Parameter("agent", agent_type)] + f.signature)
            w_fluent = Fluent("w-" + f.name, f.type, f.signature)
            self._g_fluent_map[f.name] = g_fluent
            self._l_fluent_map[f.name] = l_fluent
            self._w_fluent_map[f.name] = w_fluent
            self._new_problem.add_fluent(g_fluent, default_initial_value=False)
            self._new_problem.add_fluent(l_fluent, default_initial_value=False)
            self._new_problem.add_fluent(w_fluent, default_initial_value=False)

        for agent in self._problem.agents:
            agent_object = agent.obj
            agent.add_obj_to_problem(self._new_problem)

            end_s = InstantaneousAction("end_s_" + agent.name)
            end_s.add_precondition(Not(fin(agent_object)))
            for goal in agent.goals:                
                end_s.add_precondition(self.get_global_version(goal))
                end_s.add_precondition(self.get_local_version(goal, agent_object))
            end_s.add_effect(fin(agent_object), True)
            end_s.add_effect(act, False)
            self._new_problem.add_action(end_s)

            end_w = InstantaneousAction("end_w_" + agent.name)
            end_w.add_precondition(Not(fin(agent_object)))                
            end_w.add_precondition(waiting(agent_object))
            for goal in agent.goals:                
                end_w.add_precondition(self.get_local_version(goal, agent_object))
            end_w.add_effect(fin(agent_object), True)
            end_w.add_effect(act, False)
            self._new_problem.add_action(end_w)

            for i, goal in enumerate(agent.goals):
                end_f = InstantaneousAction("end_f_" + agent.name + "_" + str(i))
                end_f.add_precondition(Not(fin(agent_object)))
                end_f.add_precondition(Not(self.get_global_version(goal)))
                for goal in agent.goals:
                    end_f.add_precondition(self.get_local_version(goal, agent_object))
                end_f.add_effect(fin(agent_object), True)
                end_f.add_effect(act, False)
                end_f.add_effect(failure, True)
                self._new_problem.add_action(end_f)
        
        for action in self._problem.actions:
            if action.agent.name == "a":
                agent_object = action.parameters[0]
            else:
                agent_object = action.agent.obj

            # Success version - affects globals same way as original
            a_s = self.create_action_copy(action, "_s")
            a_s.add_precondition(Not(waiting(agent_object)))
            for effect in action.effects:
                if effect.value.is_true():
                    a_s.add_precondition(Not(self.get_waiting_version(effect.fluent)))
            for fact in action.preconditions + action.preconditions_wait:
                if fact.is_and():
                    for f in fact.args:
                        a_s.add_precondition(self.get_global_version(f))
                else:
                    a_s.add_precondition(self.get_global_version(fact))
            for effect in action.effects:
                a_s.add_effect(self.get_global_version(effect.fluent), effect.value)
            self._new_problem.add_action(a_s)            

            real_preconds = []
            for fact in action.preconditions:
                if fact.is_and():
                    real_preconds += fact.args
                else:
                    real_preconds.append(fact)

            # Fail version
            for i, fact in enumerate(real_preconds):
                a_f = self.create_action_copy(action, "_f_" + str(i))
                a_f.add_precondition(Not(waiting(agent_object)))
                for pre in action.preconditions_wait:
                    a_f.add_precondition(self.get_global_version(pre))
                a_f.add_precondition(Not(self.get_global_version(fact)))
                a_f.add_effect(failure, True)
                self._new_problem.add_action(a_f)

            # Wait version
            for i, fact in enumerate(action.preconditions_wait):
                a_w = self.create_action_copy(action, "_w_" + str(i))
                a_w.add_precondition(Not(waiting(agent_object)))
                a_w.add_precondition(Not(self.get_global_version(fact)))
                assert not fact.is_not()
                a_w.add_effect(self.get_waiting_version(fact), True)
                a_w.add_effect(failure, True)
                self._new_problem.add_action(a_w)

            # Phantom version
            for i, fact in enumerate(action.preconditions_wait):
                a_p = self.create_action_copy(action, "_p_" + str(i))
                a_p.add_precondition(waiting(agent_object))                
                self._new_problem.add_action(a_p)

        # Initial state
        for var, val in self._problem.initial_values.items():
            self._new_problem.set_initial_value(self.get_global_version(var), val)
            for agent in self._problem.agents:
                agent_object = unified_planning.model.Object(agent.name, agent_type)
                self._new_problem.set_initial_value(self.get_local_version(var, agent_object), val)

        # Goal
        self._new_problem.add_goal(failure)
        for agent in self._problem.agents:
            agent_object = unified_planning.model.Object(agent.name, agent_type)
            self._new_problem.add_goal(fin(agent_object))
                
        return self._new_problem
