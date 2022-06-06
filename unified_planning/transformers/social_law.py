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
"""This module defines the social law class."""

from unified_planning.transformers.transformer import Transformer
from unified_planning.transformers.single_agent_projection import SingleAgentProjection
from unified_planning.transformers.robustness_verification import RobustnessVerifier
from unified_planning.model import Parameter, Fluent, InstantaneousAction
from unified_planning.shortcuts import *
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.solvers.solver import Solver
from unified_planning.model import Problem, InstantaneousAction, DurativeAction, Action
from unified_planning.plan import Plan
from typing import List, Dict
from enum import Enum, auto
from unified_planning.io import PDDLWriter, PDDLReader


class SocialLawRobustnessStatus(Enum):
    ROBUST_RATIONAL = auto() # Social law was proven to be robust
    NON_ROBUST_SINGLE_AGENT = auto() # Social law is not robust because one of the single agent projections is unsolvable
    NON_ROBUST_MULTI_AGENT_FAIL = auto() # Social law is not robust because the compilation achieves fail
    NON_ROBUST_MULTI_AGENT_DEADLOCK = auto() # Social law is not robust because the compilation achieves a deadlock

class SocialLaw(Transformer):
    '''social law class:
    This class requires a problem (with multiple agents) and outputs a problem where the social law is applied.
    It also contains helper methods for checking robustness.    
    '''
    def __init__(self, problem: Problem, planner : Solver = None, name: str = 'sl',):
        Transformer.__init__(self, problem, name)  
        self._planner : Solver = planner
        self._counter_example : Plan = None
        self._new_problem : Problem = None

    @property
    def counter_example(self) -> Plan:
        return self._counter_example

    def get_planner(self) -> Solver:
        if self._planner is None:
            self._planner = OneshotPlanner(name='tamer')
        return self._planner

    def is_single_agent_solvable(self) -> Bool:
        problem = self.get_rewritten_problem()
        for agent in problem.agents:
            sap = SingleAgentProjection(problem, agent)
            sap_problem = sap.get_rewritten_problem()
            result = self.get_planner().solve(sap_problem)
            if result.status not in unified_planning.solvers.results.POSITIVE_OUTCOMES:
                return False
        return True        

    def is_multi_agent_robust(self) -> Bool:
        self._counter_example = None

        problem = self.get_rewritten_problem()        
        rv = RobustnessVerifier(problem)
        rv_problem = rv.get_rewritten_problem()

        w = PDDLWriter(rv_problem)
        with open("domain_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("problem_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()


        result = self.get_planner().solve(rv_problem)
        if result.status in unified_planning.solvers.results.POSITIVE_OUTCOMES:
            self._counter_example = result.plan
            return False
        return True         


    def is_robust(self) -> SocialLawRobustnessStatus:
        if not self.is_single_agent_solvable():
            return SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT
        if not self.is_multi_agent_robust():
            for action_occurence in self.counter_example.actions:
                if action_occurence.action.name[-2:] == "_f":
                    return SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
                elif action_occurence.action.name[-2:] == "_w":
                    return SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK            
            return SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL
        return SocialLawRobustnessStatus.ROBUST_RATIONAL


    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that implements the social law. Implementation here does nothing
        '''
        if self._new_problem is not None:
            return self._new_problem
        self._new_problem = self._problem
        return self._new_problem

