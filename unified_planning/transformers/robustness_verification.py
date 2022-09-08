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

from operator import neg
from unified_planning.transformers.transformer import Transformer
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from typing import List, Dict
from itertools import product


class RobustnessVerifier(Transformer):
    '''Robustness verification transformer class:
    this class requires a problem (with multiple agents, where a social law is already encoded)
    and outputs a planning problem whose solution encodes a counterexample to the robustness    
    '''

    def __init__(self, problem: Problem, name: str = 'slrob'):
        Transformer.__init__(self, problem, name)
        self._new_problem = None
        self._g_fluent_map = {}
        self._l_fluent_map = {}
        self.act_pred = None

    def get_global_version(self, fact):
        """get the global copy of given fact
        """
        # TODO: there must be a cleaner way to do this...
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

    def prepare_rewritten_problem(self):
        if self._new_problem is not None:
            return self._new_problem
        self._new_problem = Problem(f'{self._name}_{self._problem.name}')

        # Add types
        for type in self._problem.user_types:
            self._new_problem._add_user_type(type)

        # Add type for agent (if needed) and objects for agents
        self.agent_type = None
        for agent in self._problem.agents:
            if not self._new_problem.has_type(agent.obj.type.name):
                self._new_problem._add_user_type(agent.obj.type)
            if self.agent_type is None:
                self.agent_type = agent.obj.type
            else:
                # Don't know how to handle case of agents of multiple types
                assert self.agent_type == agent.obj.type

            agent.add_obj_to_problem(self._new_problem)

        # Add objects other
        self._new_problem.add_objects(self._problem.all_objects)

        # Add global and local copy for each fact
        for f in self._problem.fluents:
            g_fluent = Fluent("g-" + f.name, f.type, f.signature)
            l_fluent = Fluent("l-" + f.name, f.type, [Parameter("agent", self.agent_type)] + f.signature)
            self._g_fluent_map[f.name] = g_fluent
            self._l_fluent_map[f.name] = l_fluent
            self._new_problem.add_fluent(g_fluent, default_initial_value=False)
            self._new_problem.add_fluent(l_fluent, default_initial_value=False)


class InstantaneousActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verification transformer class:
    this class requires a problem with instantaneous actions (with multiple agents, where a social law is already encoded)
    and outputs a classical planning problem whose solution encodes a counterexample to the robustness    
    '''

    def __init__(self, problem: Problem, name: str = 'slrob'):
        RobustnessVerifier.__init__(self, problem, name)
        self._w_fluent_map = {}

    def create_action_copy(self, action, suffix):
        """Create a new copy of an action, with name action_name_suffix, and duplicates the local preconditions/effects
        """

        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = InstantaneousAction(action.name + suffix, _parameters=d)
        new_action.add_precondition(self.act_pred)
        # TODO: can probably do this better with a substitution walker
        for fact in action.preconditions + action.preconditions_wait:
            if fact.is_and():
                for f in fact.args:
                    new_action.add_precondition(self.get_local_version(f, action.agent.obj))
            else:
                new_action.add_precondition(self.get_local_version(fact, action.agent.obj))
        for effect in action.effects:
            new_action.add_effect(self.get_local_version(effect.fluent, action.agent.obj), effect.value)

        return new_action

    def get_waiting_version(self, fact):  # , agent
        """get the waiting copy of given fact
        """
        # agent_tuple = (agent),

        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        wfact = self._new_problem._env.expression_manager.FluentExp(
            self._w_fluent_map[fact.fluent().name],
            # agent_tuple +
            fact.args)
        if negate:
            return Not(wfact)
        else:
            return wfact

    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that implements the social law robustness verification compliation from Karpas, Shleyfman, Tennenholtz, ICAPS 2017 
        with the bugs fixed
        '''
        self.prepare_rewritten_problem()

        # Add fluents
        failure = Fluent("failure")
        crash = Fluent("crash")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", self.agent_type)])

        self.act_pred = act

        self._new_problem.add_fluent(failure, default_initial_value=False)
        self._new_problem.add_fluent(crash, default_initial_value=False)
        self._new_problem.add_fluent(act, default_initial_value=True)
        self._new_problem.add_fluent(fin, default_initial_value=False)
        self._new_problem.add_fluent(waiting, default_initial_value=False)

        for f in self._problem.fluents:
            w_fluent = Fluent("w-" + f.name, f.type, f.signature)
            self._w_fluent_map[f.name] = w_fluent
            self._new_problem.add_fluent(w_fluent, default_initial_value=False)

        # Add actions
        for agent in self._problem.agents:
            end_s = InstantaneousAction("end_s_" + agent.name)
            end_s.add_precondition(Not(fin(agent.obj)))
            for goal in agent.goals:
                end_s.add_precondition(self.get_global_version(goal))
                end_s.add_precondition(self.get_local_version(goal, agent.obj))
            end_s.add_effect(fin(agent.obj), True)
            end_s.add_effect(act, False)
            self._new_problem.add_action(end_s)

            # end_w = InstantaneousAction("end_w_" + agent.name)
            # end_w.add_precondition(Not(fin(agent.obj)))
            # end_w.add_precondition(waiting(agent.obj))
            # for goal in agent.goals:                
            #     end_w.add_precondition(self.get_local_version(goal, agent.obj))
            # end_w.add_effect(fin(agent.obj), True)
            # end_w.add_effect(act, False)
            # self._new_problem.add_action(end_w)

            for i, goal in enumerate(agent.goals):
                end_f = InstantaneousAction("end_f_" + agent.name + "_" + str(i))
                end_f.add_precondition(Not(fin(agent.obj)))
                end_f.add_precondition(Not(self.get_global_version(goal)))
                for g in agent.goals:
                    end_f.add_precondition(self.get_local_version(g, agent.obj))
                end_f.add_effect(fin(agent.obj), True)
                end_f.add_effect(act, False)
                end_f.add_effect(failure, True)
                self._new_problem.add_action(end_f)

        for action in self._problem.actions:
            # Success version - affects globals same way as original
            a_s = self.create_action_copy(action, "_s")
            a_s.add_precondition(Not(waiting(action.agent.obj)))
            a_s.add_precondition(Not(crash))
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
                a_f.add_precondition(Not(waiting(action.agent.obj)))
                a_f.add_precondition(Not(crash))
                for pre in action.preconditions_wait:
                    a_f.add_precondition(self.get_global_version(pre))
                a_f.add_precondition(Not(self.get_global_version(fact)))
                a_f.add_effect(failure, True)
                a_f.add_effect(crash, True)
                self._new_problem.add_action(a_f)

            # Wait version
            for i, fact in enumerate(action.preconditions_wait):
                a_w = self.create_action_copy(action, "_w_" + str(i))
                a_w.add_precondition(Not(crash))
                a_w.add_precondition(Not(waiting(action.agent.obj)))
                a_w.add_precondition(Not(self.get_global_version(fact)))
                assert not fact.is_not()
                a_w.add_effect(self.get_waiting_version(fact), True)  # , action.agent.obj), True)
                a_w.add_effect(waiting(action.agent.obj), True)
                a_w.add_effect(failure, True)
                self._new_problem.add_action(a_w)

            # Phantom version            
            a_pc = self.create_action_copy(action, "_pc")
            a_pc.add_precondition(crash)
            self._new_problem.add_action(a_pc)

            # Phantom version            
            a_pw = self.create_action_copy(action, "_pw")
            a_pw.add_precondition(waiting(action.agent.obj))
            self._new_problem.add_action(a_pw)

        # Initial state
        for var, val in self._problem.initial_values.items():
            self._new_problem.set_initial_value(self.get_global_version(var), val)
            for agent in self._problem.agents:
                self._new_problem.set_initial_value(self.get_local_version(var, agent.obj), val)

        # Goal
        self._new_problem.add_goal(failure)
        for agent in self._problem.agents:
            self._new_problem.add_goal(fin(agent.obj))

        return self._new_problem


class DuativeActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verification transformer class for durative actions:
    this class requires a problem with durative actions (with multiple agents, where a social law is already encoded)
    and outputs a temporal planning problem whose solution encodes a counterexample to the robustness    
    '''

    def __init__(self, problem: Problem, name: str = 'slrob', compile_away_numeric: bool = True,
                 max_inv_count: int = 20):
        RobustnessVerifier.__init__(self, problem, name)
        self._w_fluent_map = {}
        self._i_fluent_map = {}
        self._compile_away_numeric = compile_away_numeric
        self._max_inv_count = max_inv_count

    def create_action_copy(self, action, suffix):
        """Create a new copy of an action, with name action_name_suffix, and duplicates the local preconditions/effects
        """

        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = DurativeAction(action.name + suffix, _parameters=d)
        new_action.set_duration_constraint(action.duration)
        new_action.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), self.act_pred)

        # TODO: can probably do this better with a substitution walker
        for timing in action.conditions.keys():
            for fact in action.conditions[timing]:
                if fact.is_and():
                    for f in fact.args:
                        new_action.add_condition(timing, self.get_local_version(f, action.agent.obj))
                else:
                    new_action.add_condition(timing, self.get_local_version(fact, action.agent.obj))
        for timing in action.conditions_wait.keys():
            for fact in action.conditions_wait[timing]:
                if fact.is_and():
                    for f in fact.args:
                        new_action.add_condition(timing, self.get_local_version(f, action.agent.obj))
                else:
                    new_action.add_condition(timing, self.get_local_version(fact, action.agent.obj))
        for timing in action.effects.keys():
            for effect in action.effects[timing]:
                new_action.add_effect(timing, self.get_local_version(effect.fluent, action.agent.obj), effect.value)

        return new_action

    def get_waiting_version(self, fact, agent):
        """get the waiting copy of given fact
        """
        agent_tuple = (agent),

        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        wfact = self._new_problem._env.expression_manager.FluentExp(
            self._w_fluent_map[fact.fluent().name],
            agent_tuple + fact.args)
        if negate:
            return Not(wfact)
        else:
            return wfact

    def get_inv_count_version(self, fact, val=None):
        """get the invariant count copy of given fact
        """
        if self._compile_away_numeric:
            val_tuple = (Object("ic" + str(val), self.count_type)),
            new_args = fact.args + val_tuple
            ifact = self._new_problem._env.expression_manager.FluentExp(self._i_fluent_map[fact.fluent().name],
                                                                        new_args)
        else:
            ifact = self._new_problem._env.expression_manager.FluentExp(self._i_fluent_map[fact.fluent().name],
                                                                        fact.args)
        return ifact

    def add_increase_inv_count_version(self, fact, action: DurativeAction):
        """ Add to action an effect which increases the inv_count of the given fact"""
        return
        if self._compile_away_numeric:
            for i in range(self._max_inv_count - 1):
                action.add_effect(StartTiming(), self.get_inv_count_version(fact, i + 1), True,
                                  self.get_inv_count_version(fact, i))
                action.add_effect(StartTiming(), self.get_inv_count_version(fact, i), False,
                                  self.get_inv_count_version(fact, i))
        else:
            action.add_increase_effect(StartTiming(), self.get_inv_count_version(fact), 1)

    def add_decrease_inv_count_version(self, fact, action: DurativeAction):
        """ Add to action an effect which increases the inv_count of the given fact"""
        return
        if self._compile_away_numeric:
            for i in range(1, self._max_inv_count):
                action.add_effect(StartTiming(), self.get_inv_count_version(fact, i), True,
                                  self.get_inv_count_version(fact, i - 1))
                action.add_effect(StartTiming(), self.get_inv_count_version(fact, i), False,
                                  self.get_inv_count_version(fact, i))
        else:
            action.add_decrease_effect(EndTiming(), self.get_inv_count_version(fact), 1)

    def add_condition_inv_count_zero(self, fact, action: DurativeAction, timing: Timing, negate: bool, val: int):
        """ Add a condition which checks that no one is waiting for the given invariant fact"""
        return
        if self._compile_away_numeric:
            bcond = self.get_inv_count_version(fact, val)
            if negate:
                cond = Not(bcond)
            else:
                cond = bcond

        else:
            if negate:
                cond = GT(self.get_inv_count_version(fact), val)
            else:
                cond = Equals(self.get_inv_count_version(fact), val)

        action.add_condition(timing, cond)

    def get_conditions_at(self, conditions):
        c_start = []
        c_overall = []
        c_end = []
        for interval, cl in conditions.items():
            for c in cl:
                if interval.lower == interval.upper:
                    if interval.lower.is_from_start():
                        c_start.append(c)
                    else:
                        c_end.append(c)
                else:
                    if not interval.is_left_open():
                        c_start.append(c)
                    c_overall.append(c)
                    if not interval.is_right_open():
                        c_end.append(c)
        return (c_start, c_overall, c_end)

    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that implements the social law robustness verification compliation from Karpas, Shleyfman, Tennenholtz, ICAPS 2017 
        with the bugs fixed
        '''
        self.prepare_rewritten_problem()

        if self._compile_away_numeric:
            self.count_type = UserType("invcount_t")
            self._new_problem._add_user_type(self.count_type)
            self.icnext = Fluent("icnext",
                                 _signature=[Parameter("ic", self.count_type), Parameter("icnext", self.count_type)])
            self._new_problem.add_fluent(self.icnext, default_initial_value=False)

            for i in range(self._max_inv_count):
                self._new_problem.add_object(Object("ic" + str(i), self.count_type))
                if i + 1 < self._max_inv_count:
                    self._new_problem.set_initial_value(
                        self.icnext(Object("ic" + str(i), self.count_type), Object("ic" + str(i + 1), self.count_type)),
                        True)

        # Add fluents
        failure = Fluent("failure")
        act = Fluent("act")
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])
        waiting = Fluent("waiting", _signature=[Parameter("a", self.agent_type)])

        self.act_pred = act

        self._new_problem.add_fluent(failure, default_initial_value=False)
        self._new_problem.add_fluent(act, default_initial_value=True)
        self._new_problem.add_fluent(fin, default_initial_value=False)
        self._new_problem.add_fluent(waiting, default_initial_value=False)

        for f in self._problem.fluents:
            w_fluent = Fluent("w-" + f.name, f.type, [Parameter("agent", self.agent_type)] + f.signature)
            self._w_fluent_map[f.name] = w_fluent
            self._new_problem.add_fluent(w_fluent, default_initial_value=False)

            if self._compile_away_numeric:
                i_fluent = Fluent("i-" + f.name, _signature=f.signature + [Parameter("invcount", self.count_type)])
                self._i_fluent_map[f.name] = i_fluent
                self._new_problem.add_fluent(i_fluent, default_initial_value=False)
            else:
                i_fluent = Fluent("i-" + f.name, IntType(), f.signature)
                self._i_fluent_map[f.name] = i_fluent
                self._new_problem.add_fluent(i_fluent, default_initial_value=0)

        for action in self._problem.actions:
            # Success version - affects globals same way as original
            a_s = self.create_action_copy(action, "_s")
            a_s.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
            for timing in action.conditions.keys():
                for fact in action.conditions[timing]:
                    if fact.is_and():
                        for f in fact.args:
                            a_s.add_condition(timing, self.get_global_version(f))
                    else:
                        a_s.add_condition(timing, self.get_global_version(fact))
            for timing in action.conditions_wait.keys():
                for fact in action.conditions_wait[timing]:
                    if fact.is_and():
                        for f in fact.args:
                            a_s.add_condition(timing, self.get_global_version(f))
                    else:
                        a_s.add_condition(timing, self.get_global_version(fact))
            for timing in action.effects.keys():
                for effect in action.effects[timing]:
                    a_s.add_effect(timing, self.get_global_version(effect.fluent), effect.value)

            for timing in [StartTiming(), EndTiming()]:
                for effect in action.effects.get(timing, []):
                    if effect.value.is_false():
                        if timing == StartTiming():
                            self.add_condition_inv_count_zero(effect.fluent, a_s, timing, False, 0)
                        else:
                            self.add_condition_inv_count_zero(effect.fluent, a_s, timing, False, 1)
                    if effect.value.is_true():
                        for agent in self._problem.agents:
                            a_s.add_condition(timing, Not(self.get_waiting_version(effect.fluent, agent.obj)))
                            if timing == EndTiming():
                                a_s.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()),
                                                  Not(self.get_waiting_version(effect.fluent, agent.obj)))
            for interval, condition in action.conditions.items():
                if interval.lower != interval.upper:
                    for fact in condition:
                        self.add_increase_inv_count_version(fact, a_s)
                        self.add_decrease_inv_count_version(fact, a_s)
            self._new_problem.add_action(a_s)

            c_start, c_overall, c_end = self.get_conditions_at(action.conditions)
            cw_start, cw_overall, cw_end = self.get_conditions_at(action.conditions_wait)

            # Fail start version            
            for i, fact in enumerate(c_start):
                a_fstart = self.create_action_copy(action, "_f_start_" + str(i))
                for c in cw_start:
                    a_fstart.add_condition(StartTiming(), self.get_global_version(c))
                a_fstart.add_condition(StartTiming(), Not(self.get_global_version(fact)))
                a_fstart.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                a_fstart.add_effect(StartTiming(), failure, True)
                self._new_problem.add_action(a_fstart)

            # Fail inv version            
            for i, fact in enumerate(c_overall):
                overall_condition_added_by_start_effect = False
                for effect in action.effects.get(StartTiming(), []):
                    if effect.fluent == fact and effect.value.is_true():
                        overall_condition_added_by_start_effect = True
                        break
                if not overall_condition_added_by_start_effect:
                    a_finv = self.create_action_copy(action, "_f_inv_" + str(i))
                    for c in c_start + cw_start:
                        a_fstart.add_condition(StartTiming(), self.get_global_version(c))
                    a_finv.add_condition(StartTiming(), Not(self.get_global_version(fact)))
                    for effect in action.effects.get(StartTiming(), []):
                        a_finv.add_effect(StartTiming(), self.get_global_version(effect.fluent), effect.value)
                        if effect.value.is_false():
                            self.add_condition_inv_count_zero(effect.fluent, a_finv, StartTiming(), False, 0)
                        if effect.value.is_true():
                            for agent in self._problem.agents:
                                a_finv.add_condition(StartTiming(),
                                                     Not(self.get_waiting_version(effect.fluent, agent.obj)))
                    a_finv.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                    a_finv.add_effect(StartTiming(), failure, True)
                    self._new_problem.add_action(a_finv)

            # Fail end version            
            for i, fact in enumerate(c_end):
                a_fend = self.create_action_copy(action, "_f_end_" + str(i))
                for c in c_start + cw_start:
                    a_fend.add_condition(StartTiming(), self.get_global_version(c))
                for c in c_overall:
                    a_fend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()), self.get_global_version(c))
                a_fend.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                a_fend.add_condition(EndTiming(), Not(self.get_global_version(fact)))
                for effect in action.effects.get(StartTiming(), []):
                    a_fend.add_effect(StartTiming(), self.get_global_version(effect.fluent), effect.value)
                    if effect.value.is_false():
                        self.add_condition_inv_count_zero(effect.fluent, a_fend, StartTiming(), False, 0)
                    if effect.value.is_true():
                        for agent in self._problem.agents:
                            a_fend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),
                                                 Not(self.get_waiting_version(effect.fluent, agent.obj)))
                for effect in action.effects.get(EndTiming(), []):
                    # if effect.value.is_false():
                    #    a_fend.add_condition(StartTiming(), Equals(self.get_inv_count_version(effect.fluent), 0))
                    if effect.value.is_true():
                        for agent in self._problem.agents:
                            a_fend.add_condition(StartTiming(), Not(self.get_waiting_version(effect.fluent, agent.obj)))
                a_fend.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                a_fend.add_effect(EndTiming(), failure, True)
                self.add_increase_inv_count_version(fact, a_fend)
                self._new_problem.add_action(a_fend)

            # Del inv start version            
            for i, effect in enumerate(action.effects.get(StartTiming(), [])):
                if effect.value.is_false():
                    a_finvstart = self.create_action_copy(action, "_f_inv_start_" + str(i))
                    a_finvstart.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                    for c in c_start + cw_start:
                        a_fend.add_condition(StartTiming(), self.get_global_version(c))
                    self.add_condition_inv_count_zero(effect.fluent, a_finvstart, StartTiming(), True, 0)
                    a_finvstart.add_effect(StartTiming(), failure, True)
                    self._new_problem.add_action(a_finvstart)

            # Del inv end version            
            for i, effect in enumerate(action.effects.get(EndTiming(), [])):
                if effect.value.is_false():
                    a_finvend = self.create_action_copy(action, "_f_inv_end_" + str(i))
                    a_finvend.add_condition(StartTiming(), Not(waiting(action.agent.obj)))
                    for c in c_start + cw_start:
                        a_finvend.add_condition(StartTiming(), self.get_global_version(c))
                    for c in c_overall:
                        a_finvend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),
                                                self.get_global_version(c))
                    for c in c_end:
                        a_finvend.add_condition(EndTiming(), self.get_global_version(c))
                    self.add_condition_inv_count_zero(effect.fluent, a_finvend, EndTiming(), True, 0)

                    for seffect in action.effects.get(StartTiming(), []):
                        a_finvend.add_effect(StartTiming(), self.get_global_version(seffect.fluent), seffect.value)
                        if seffect.value.is_false():
                            self.add_condition_inv_count_zero(seffect.fluent, a_finvend, StartTiming(), False, 0)
                        if seffect.value.is_true():
                            for agent in self._problem.agents:
                                a_finvend.add_condition(StartTiming(),
                                                        Not(self.get_waiting_version(seffect.fluent, agent.obj)))
                                a_finvend.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),
                                                        Not(self.get_waiting_version(seffect.fluent, agent.obj)))
                    for seffect in action.effects.get(EndTiming(), []):
                        a_finvend.add_effect(EndTiming(), self.get_global_version(seffect.fluent), seffect.value)

                    # self.add_condition_inv_count_zero(effect.fluent, a_finvend, StartTiming(), True, 0)
                    a_finvend.add_effect(StartTiming(), failure, True)
                    for interval, condition in action.conditions.items():
                        if interval.lower != interval.upper:
                            for fact in condition:
                                self.add_increase_inv_count_version(fact, a_finvend)
                    self._new_problem.add_action(a_finvend)

            # a^w_x version - wait forever for x to be true
            for w_fact in cw_start:
                a_wx = self.create_action_copy(action, "_wx_" + f.name)
                a_wx.add_condition(StartTiming(), Not(waiting(action.agent.obj)))

                a_wx.add_effect(StartTiming(), failure, True)
                a_wx.add_effect(StartTiming(), waiting(action.agent.obj), True)
                a_wx.add_condition(StartTiming(), Not(self.get_global_version(w_fact)))
                a_wx.add_effect(StartTiming(), self.get_waiting_version(w_fact, action.agent.obj), True)
                self._new_problem.add_action(a_wx)

            # a_waiting version - dummy version while agent is waiting
            a_waiting = self.create_action_copy(action, "_waiting")
            a_waiting.add_condition(StartTiming(), waiting(action.agent.obj))
            self._new_problem.add_action(a_waiting)

        for i, agent in enumerate(self._problem.agents):
            # Create end_s_i action
            end_s_action = InstantaneousAction("end_s_" + str(i))
            end_s_action.add_precondition(Not(fin(agent.obj)))
            for g in agent.goals:
                end_s_action.add_precondition(self.get_local_version(g, agent.obj))
                end_s_action.add_precondition(self.get_global_version(g))
            end_s_action.add_effect(fin(agent.obj), True)
            end_s_action.add_effect(act, False)
            self._new_problem.add_action(end_s_action)

            # Create end_f_i action
            for j, gf in enumerate(agent.goals):
                end_f_action = InstantaneousAction("end_f_" + str(i) + "_" + str(j))
                end_f_action.add_precondition(Not(self.get_global_version(gf)))
                end_f_action.add_precondition(Not(fin(agent.obj)))
                for g in agent.goals:
                    end_f_action.add_precondition(self.get_local_version(g, agent.obj))
                end_f_action.add_effect(fin(agent.obj), True)
                end_f_action.add_effect(failure, True)
                end_f_action.add_effect(act, False)
                self._new_problem.add_action(end_f_action)

        # Initial state
        for var, val in self._problem.initial_values.items():
            self._new_problem.set_initial_value(self.get_global_version(var), val)
            for agent in self._problem.agents:
                self._new_problem.set_initial_value(self.get_local_version(var, agent.obj), val)

        # Goal
        self._new_problem.add_goal(failure)
        for agent in self._problem.agents:
            self._new_problem.add_goal(fin(agent.obj))

        return self._new_problem


























class WaitingActionRobustnessVerifier(RobustnessVerifier):
    '''Robustness verification transformer class:
    this class requires a problem with instantaneous actions (with multiple agents, where a social law is already encoded)
    and outputs a classical planning problem whose solution encodes a counterexample to the robustness
    '''

    def __init__(self, problem: Problem, name: str = 'slrob'):
        RobustnessVerifier.__init__(self, problem, name)
        self._w_fluent_map = {}

    def create_action_copy(self, action, suffix):
        """Create a new copy of an action, with name action_name_suffix, and duplicates the local preconditions/effects
        """

        d = {}
        for p in action.parameters:
            d[p.name] = p.type

        new_action = InstantaneousAction(action.name + suffix, _parameters=d)
        # new_action.add_precondition(self.act_pred)
        # TODO: can probably do this better with a substitution walker
        for fact in action.preconditions + action.preconditions_wait:
            if fact.is_and():
                for f in fact.args:
                    new_action.add_precondition(self.get_local_version(f, action.agent.obj))
            else:
                new_action.add_precondition(self.get_local_version(fact, action.agent.obj))
        for effect in action.effects:
            new_action.add_effect(self.get_local_version(effect.fluent, action.agent.obj), effect.value)

        return new_action

    def get_waiting_version(self, fact):  # , agent
        """get the waiting copy of given fact
        """
        # agent_tuple = (agent),

        negate = False
        if fact.is_not():
            negate = True
            fact = fact.arg(0)
        wfact = self._new_problem._env.expression_manager.FluentExp(
            self._w_fluent_map[fact.fluent().name],
            # agent_tuple +
            fact.args)
        if negate:
            return Not(wfact)
        else:
            return wfact

    def get_rewritten_problem(self) -> Problem:
        """
        Creates a problem that implements the social law robustness verification compilation from Tuisov, Shleyfman, Karpas
        with the bugs fixed
        """
        self.prepare_rewritten_problem()

        # Add fluents
        aux = stage_1, stage_2, precondition_violation, possible_deadlock, conflict = (Fluent("stage 1"),
                                                                                       Fluent("stage 2"),
                                                                                       Fluent("precondition violation"),
                                                                                       Fluent("possible deadlock"),
                                                                                       Fluent("conflict"))
        fin = Fluent("fin", _signature=[Parameter("a", self.agent_type)])

        # self._new_problem.add_fluent(Fluent("stage 1"), default_initial_value=True)
        for fluent in aux:
            self._new_problem.add_fluent(fluent, default_initial_value=False)

        allow_action_map = {}
        for action in self._problem.actions:
            action_fluent = Fluent("allow-" + action.name)
            # allow_action_map.setdefault(action.agent, {}).update(action=action_fluent)
            if action.agent.name not in allow_action_map.keys():
                allow_action_map[action.agent.name] = {action.name: action_fluent}
            else:
                allow_action_map[action.agent.name][action.name] = action_fluent
            self._new_problem.add_fluent(action_fluent, default_initial_value=True)

        # Add actions
        for action in self._problem.actions:
            agent = action.agent
            # Success version - affects globals same way as original
            a_s = self.create_action_copy(action, "_s")
            a_s.add_precondition(stage_1)
            a_s.add_precondition(allow_action_map[action.agent.name][action.name])
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
                a_f.add_precondition(stage_1)
                a_f.add_precondition(allow_action_map[action.agent.name][action.name])
                for pre in action.preconditions_wait:
                    a_f.add_precondition(self.get_global_version(pre))
                a_f.add_precondition(Not(self.get_global_version(fact)))
                a_f.add_effect(precondition_violation, True)
                a_f.add_effect(stage_2, True)
                a_f.add_effect(stage_1, False)

                self._new_problem.add_action(a_f)

            for i, fact in enumerate(action.preconditions_wait):
                # Wait version
                a_w = self.create_action_copy(action, "_w_" + str(i))
                a_s.add_precondition(stage_1)
                a_s.add_precondition(allow_action_map[action.agent.name][action.name])
                a_w.add_precondition(Not(self.get_global_version(fact)))
                assert not fact.is_not()
                a_w.add_effect(self.get_waiting_version(fact), True)  # , action.agent.obj), True)
                self._new_problem.add_action(a_w)

                # deadlock version
                a_deadlock = self.create_action_copy(action, "_deadlock_" + str(i))
                a_deadlock.add_precondition(Not(self.get_global_version(fact)))
                for another_action in allow_action_map[agent].keys():
                    a_deadlock.add_precondition(Not(allow_action_map[agent][another_action]))
                a_deadlock.add_effect(fin(agent), True)
                a_deadlock.add_effect(possible_deadlock, True)

            # local version
            a_local = self.create_action_copy(action, "_local")
            a_local.add_precondition(stage_2)
            a_local.add_precondition(allow_action_map[action.agent.name][action.name])
            for fluent in allow_action_map[action.agent.name].values():
                print(fluent)
                a_local.add_effect(fluent, True)

            self._new_problem.add_action(a_local)

        #end-success
        for agent in self._problem.agents:
            end_s = InstantaneousAction("end_s_" + agent.name)
            for goal in agent.goals:
                end_s.add_precondition(self.get_global_version(goal))
                end_s.add_precondition(self.get_local_version(goal, agent.obj))
            end_s.add_effect(fin(agent.obj), True)
            end_s.add_effect(stage_1, False)
            self._new_problem.add_action(end_s)

        # start-stage-2
        start_stage_2 = InstantaneousAction("start_stage_2")
        for agent in self._problem.agents:
            start_stage_2.add_precondition(fin(agent.obj))
        start_stage_2.add_effect(stage_2, True)
        start_stage_2.add_effect(stage_1, False)
        self._new_problem.add_action(start_stage_2)

        # goals_not_achieved
        goals_not_achieved = InstantaneousAction("goals_not_achieved")
        goals_not_achieved.add_precondition(stage_2)
        for agent in self._problem.agents:
            for i, goal in enumerate(agent.goals):
                goals_not_achieved.add_precondition(Not(self.get_global_version(goal)))
                for g in agent.goals:
                    goals_not_achieved.add_precondition(self.get_local_version(g, agent.obj))
        goals_not_achieved.add_effect(conflict, True)
        self._new_problem.add_action(goals_not_achieved)


        # declare_deadlock
        declare_deadlock = InstantaneousAction("goals_not_achieved")
        declare_deadlock.add_precondition(stage_2)
        declare_deadlock.add_precondition(possible_deadlock)
        for agent in self._problem.agents:
            for i, goal in enumerate(agent.goals):
                for g in agent.goals:
                    declare_deadlock.add_precondition(self.get_local_version(g, agent.obj))
        declare_deadlock.add_effect(conflict, True)
        self._new_problem.add_action(declare_deadlock)

        # declare_fail
        declare_fail = InstantaneousAction("goals_not_achieved")
        declare_fail.add_precondition(stage_2)
        declare_fail.add_precondition(precondition_violation)
        for agent in self._problem.agents:
            for i, goal in enumerate(agent.goals):
                for g in agent.goals:
                    declare_fail.add_precondition(self.get_local_version(g, agent.obj))
        declare_fail.add_effect(conflict, True)
        self._new_problem.add_action(declare_fail)

        # Initial state
        for var, val in self._problem.initial_values.items():
            self._new_problem.set_initial_value(self.get_global_version(var), val)
            for agent in self._problem.agents:
                self._new_problem.set_initial_value(self.get_local_version(var, agent.obj), val)

        # Goal
        self._new_problem.add_goal(conflict)

        return self._new_problem
