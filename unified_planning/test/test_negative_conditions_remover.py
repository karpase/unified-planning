# Copyright 2021 AIPlan4EU project
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


from fractions import Fraction
import os
from unified_planning.plan import ActionInstance
import unified_planning
from unified_planning.environment import get_env
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main, skipIfNoPlanValidatorForProblemKind, skipIfNoOneshotPlannerForProblemKind
from unified_planning.test.examples import get_example_problems
from unified_planning.model.problem_kind import basic_classical_kind, classical_kind, basic_temporal_kind, full_classical_kind
from unified_planning.transformers import NegativeConditionsRemover
from unified_planning.solvers import SequentialPlanValidator as PV
from unified_planning.exceptions import UPExpressionDefinitionError, UPProblemDefinitionError


class TestNegativeConditionsRemover(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.env = get_env()
        self.problems = get_example_problems()

    @skipIfNoOneshotPlannerForProblemKind(basic_classical_kind)
    @skipIfNoPlanValidatorForProblemKind(classical_kind)
    def test_basic(self):
        problem = self.problems['basic'].problem
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        self.assertEqual(len(problem.fluents) + 1, len(positive_problem.fluents))
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        with OneshotPlanner(problem_kind=positive_problem.kind) as planner:
            self.assertNotEqual(planner, None)
            positive_plan = planner.solve(positive_problem).plan
            new_plan = npr.rewrite_back_plan(positive_plan)
            with PlanValidator(problem_kind=problem.kind) as PV:
                self.assertTrue(PV.validate(problem, new_plan))

    @skipIfNoOneshotPlannerForProblemKind(basic_classical_kind)
    @skipIfNoPlanValidatorForProblemKind(classical_kind)
    def test_robot_loader_mod(self):
        problem = self.problems['robot_loader_mod'].problem
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        positive_problem_2 = npr.get_rewritten_problem()
        self.assertEqual(positive_problem, positive_problem_2)
        self.assertEqual(len(problem.fluents) + 4, len(positive_problem.fluents))
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        with OneshotPlanner(problem_kind=positive_problem.kind) as planner:
            self.assertNotEqual(planner, None)
            positive_plan = planner.solve(positive_problem).plan
            new_plan = npr.rewrite_back_plan(positive_plan)
            with PlanValidator(problem_kind=problem.kind) as PV:
                self.assertTrue(PV.validate(problem, new_plan))

    @skipIfNoOneshotPlannerForProblemKind(basic_classical_kind.union(basic_temporal_kind))
    @skipIfNoPlanValidatorForProblemKind(classical_kind.union(basic_temporal_kind))
    def test_matchcellar(self):
        problem = self.problems['matchcellar'].problem
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        with OneshotPlanner(problem_kind=positive_problem.kind) as planner:
            self.assertNotEqual(planner, None)
            positive_plan = planner.solve(positive_problem).plan
            new_plan = npr.rewrite_back_plan(positive_plan)
            with PlanValidator(problem_kind=problem.kind) as PV:
                self.assertTrue(PV.validate(problem, new_plan))
        self.assertEqual(len(problem.fluents) + 1, len(positive_problem.fluents))
        light_match = problem.action('light_match')
        mend_fuse = problem.action('mend_fuse')
        m1 = problem.object('m1')
        m2 = problem.object('m2')
        m3 = problem.object('m3')
        f1 = problem.object('f1')
        f2 = problem.object('f2')
        f3 = problem.object('f3')
        light_m1 = ActionInstance(light_match, (ObjectExp(m1), ))
        light_m2 = ActionInstance(light_match, (ObjectExp(m2), ))
        light_m3 = ActionInstance(light_match, (ObjectExp(m3), ))
        mend_f1 = ActionInstance(mend_fuse, (ObjectExp(f1), ))
        mend_f2 = ActionInstance(mend_fuse, (ObjectExp(f2), ))
        mend_f3 = ActionInstance(mend_fuse, (ObjectExp(f3), ))
        npa = [a for s, a, d in new_plan.actions]
        self.assertIn(light_m1, npa)
        self.assertIn(light_m2, npa)
        self.assertIn(light_m3, npa)
        self.assertIn(mend_f1, npa)
        self.assertIn(mend_f2, npa)
        self.assertIn(mend_f3, npa)


    @skipIfNoOneshotPlannerForProblemKind(full_classical_kind)
    @skipIfNoPlanValidatorForProblemKind(full_classical_kind)
    def test_basic_conditional(self):
        problem = self.problems['basic_conditional'].problem
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        self.assertEqual(len(problem.fluents) + 2, len(positive_problem.fluents))
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        with OneshotPlanner(problem_kind=positive_problem.kind) as planner:
            self.assertNotEqual(planner, None)
            positive_plan = planner.solve(positive_problem).plan
            new_plan = npr.rewrite_back_plan(positive_plan)
            with PlanValidator(problem_kind=problem.kind) as PV:
                self.assertTrue(PV.validate(problem, new_plan))

    def test_temporal_conditional(self):
        problem = self.problems['temporal_conditional'].problem
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        self.assertEqual(len(problem.fluents) + 3, len(positive_problem.fluents))
        self.assertTrue(problem.kind.has_negative_conditions())
        self.assertFalse(positive_problem.kind.has_negative_conditions())
        set_giver = problem.action('set_giver')
        new_actions = npr.get_transformed_actions(set_giver)
        self.assertEqual(len(new_actions), 1)
        self.assertNotEqual(new_actions[0], set_giver)
        take_ok = problem.action('take_ok')
        new_actions = npr.get_transformed_actions(take_ok)
        self.assertEqual(len(new_actions), 1)
        self.assertNotEqual(new_actions[0], take_ok)
        #lacking planners to test this with planner+plan_validator

    def test_ad_hoc_1(self):
        x = Fluent('x')
        y = Fluent('y')
        a = InstantaneousAction('a')
        a.add_precondition(And(Not(x), Not(y)))
        a.add_effect(x, True)
        problem = Problem('ad_hoc')
        problem.add_fluent(x)
        problem.add_fluent(y)
        problem.add_action(a)
        problem.set_initial_value(x, False)
        problem.set_initial_value(y, False)
        problem.add_goal(x)
        problem.add_goal(Not(y))
        problem.add_goal(Not(Iff(x, y)))
        problem.add_timed_goal(GlobalStartTiming(5), x)
        problem.add_timed_goal(ClosedTimeInterval(GlobalStartTiming(3), GlobalStartTiming(4)), x)
        npr = NegativeConditionsRemover(problem)
        with self.assertRaises(UPExpressionDefinitionError) as e:
            positive_problem = npr.get_rewritten_problem()
        self.assertIn(f"Expression: {Not(Iff(x, y))} is not in NNF.", str(e.exception))

    def test_ad_hoc_2(self):
        x = Fluent('x')
        y = Fluent('y')
        t = GlobalStartTiming(5)
        problem = Problem('ad_hoc')
        problem.add_fluent(x)
        problem.add_fluent(y)
        problem.add_timed_effect(t, y, x, Not(y))
        problem.set_initial_value(x, True)
        problem.set_initial_value(y, False)
        problem.add_goal(x)
        npr = NegativeConditionsRemover(problem)
        positive_problem = npr.get_rewritten_problem()
        self.assertEqual(len(problem.fluents) + 1, len(positive_problem.fluents))
        y__negated__ = Fluent('ncrm_y_0')
        test_problem = Problem(positive_problem.name)
        test_problem.add_fluent(x)
        test_problem.add_fluent(y)
        test_problem.add_fluent(y__negated__)
        test_problem.add_timed_effect(t, y, x, y__negated__)
        test_problem.add_timed_effect(t, y__negated__, Not(x), y__negated__)
        test_problem.set_initial_value(x, True)
        test_problem.set_initial_value(y, False)
        test_problem.set_initial_value(y__negated__, True)
        test_problem.add_goal(x)
        self.assertEqual(positive_problem, test_problem)
