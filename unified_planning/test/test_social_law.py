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
# limitations under the License

import os
import tempfile
from typing import cast
import pytest
import unified_planning
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.model import Agent
from unified_planning.transformers import RobustnessVerifier


# (define (domain intersection)
 
#  (:requirements :strips :typing :multi-agent :unfactored-privacy)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DOMAINS_PATH = os.path.join(FILE_PATH, 'pddl')

class TestSocialLaws(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    def create_basic_intersection_problem_interface(self) -> unified_planning.model.Problem:
        #  (:types
        #   direction loc agent - object
        #   car - agent            
        #  )        
        loc = UserType("loc")
        direction = UserType("direction")
        car = UserType("car")

        #  (:predicates 
        #   (at ?a - car ?l - loc)  
        #   (free ?l - loc)  
        
        #   (arrived ?a - car)
        #   (start ?a - car ?l - loc)
        #   (travel-direction ?a - car ?d - direction)
        
        #   (connected ?l1 - loc ?l2 - loc ?d - direction)
        #   (yields-to ?l1 - loc ?l2 - loc)      
        #  )
        at = Fluent('at', _signature=[Parameter('a', car), Parameter('l', loc)])
        free = Fluent('free', _signature=[Parameter('l', loc)])
        arrived = Fluent('arrived', _signature=[Parameter('a', car)])
        start = Fluent('start', _signature=[Parameter('a', car), Parameter('l', loc)])        
        traveldirection = Fluent('traveldirection', _signature=[Parameter('a', car), Parameter('d', direction)])
        connected = Fluent('connected', _signature=[Parameter('l1', loc), Parameter('l2', loc), Parameter('d', direction)])
        #yieldsto = Fluent('traveldirection', _signature=[Parameter('l1', loc), Parameter('l2', loc)])


        #  (:constants
        #    south-ent south-ex north-ent north-ex east-ent  east-ex west-ent  west-ex cross-nw cross-ne cross-se cross-sw dummy - loc
        #    north south east west - direction
        #  )

        intersection_map = {
            "north": ["south-ent", "cross-se", "cross-ne", "north-ex"],
            "south": ["north-ent", "cross-nw", "cross-sw", "south-ex"],
            "west": ["east-ent", "cross-ne", "cross-nw", "west-ex"],
            "east": ["west-ent", "cross-sw", "cross-se", "east-ex"]
        }

        location_names = set()
        
        for l in intersection_map.values():
            location_names = location_names.union(l)
        locations = list(map(lambda l: unified_planning.model.Object(l, loc), location_names))
        self.assertEqual(len(location_names), 12)
            
        direction_names = intersection_map.keys()
        directions = list(map(lambda d: unified_planning.model.Object(d, direction), direction_names))
        
        #  (:action arrive
        #     :agent    ?a - car 
        #     :parameters  (?l - loc)
        #     :precondition  (and  
        #     	(start ?a ?l)
        #     	(not (arrived ?a))
        #     	(free ?l)      
        #       )
        #     :effect    (and     	
        #     	(at ?a ?l)
        #     	(not (free ?l))
        #     	(arrived ?a)
        #       )
        #   )
        arrive = InstantaneousAction('arrive', a=car, l=loc)
        a = arrive.parameter('a')
        l = arrive.parameter('l')
        arrive.add_precondition(start(a, l))
        arrive.add_precondition(Not(arrived(a)))
        arrive.add_precondition(free(l))
        arrive.add_effect(at(a,l), True)
        arrive.add_effect(free(l), False)
        arrive.add_effect(arrived(a), True)


        #   (:action drive
        #     :agent    ?a - car 
        #     :parameters  (?l1 - loc ?l2 - loc ?d - direction ?ly - loc)
        #     :precondition  (and      	
        #     	(at ?a ?l1)
        #     	(free ?l2)     
        #     	(travel-direction ?a ?d)
        #     	(connected ?l1 ?l2 ?d)
        #     	(yields-to ?l1 ?ly)
        #     	(free ?ly)
        #       )
        #     :effect    (and     	
        #     	(at ?a ?l2)
        #     	(not (free ?l2))
        #     	(not (at ?a ?l1))
        #     	(free ?l1)
        #       )
        #    )    
        # )
        drive = InstantaneousAction('drive', a=car, l1=loc, l2=loc, d=direction, ly=loc)
        a = drive.parameter('a')
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')
        ly = drive.parameter('ly')
        drive.add_precondition(at(a,l1))
        drive.add_precondition(free(l2))
        drive.add_precondition(traveldirection(a,d))
        drive.add_precondition(connected(l1,l2,d))
        #drive.add_precondition(yieldsto(l1,ly))
        #drive.add_precondition(free(ly))
        drive.add_effect(at(a,l2),True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(a,l1), False)
        drive.add_effect(free(l1), True)

        problem = Problem('intersection')
        problem.add_fluent(at, default_initial_value=False)
        problem.add_fluent(free, default_initial_value=True)
        problem.add_fluent(start, default_initial_value=False)
        problem.add_fluent(arrived, default_initial_value=False)
        problem.add_fluent(traveldirection, default_initial_value=False)
        problem.add_fluent(connected, default_initial_value=False)
        #problem.add_fluent(yieldsto)        

        problem.add_action(arrive)
        problem.add_action(drive)

        problem.add_objects(locations)
        problem.add_objects(directions)

        #for l in locations:
        #    problem.set_initial_value(free(unified_planning.model.Object(l, loc)), True)
        for d in intersection_map.keys():
            path = intersection_map[d]
            for i in range(len(path) - 1):
                problem.set_initial_value(connected(
                    unified_planning.model.Object(path[i], loc),
                    unified_planning.model.Object(path[i+1], loc),
                    unified_planning.model.Object(d, direction)), True)
        
        return problem


    def add_car(self, problem : unified_planning.model.Problem, name : str , startloc : str, endloc : str, cardirection : str):
        cartype = problem.user_type("car")
        loc = problem.user_type("loc")
        direction = problem.user_type("direction")

        carobj = unified_planning.model.Object(name, cartype)
        problem.add_objects([carobj])

        start = problem.fluent("start")
        problem.set_initial_value(start(carobj, unified_planning.model.Object(startloc, loc)), True)

        traveldirection = problem.fluent("traveldirection")
        problem.set_initial_value(traveldirection(carobj, unified_planning.model.Object(cardirection, direction)), True)

        at = problem.fluent("at")
        cargoal = at(carobj, unified_planning.model.Object(endloc, loc))

        #caragent = Agent("agent-" + name, problem.env)
        #caragent.add_goal(cargoal)
        #problem.add_agent(caragent)
        problem.add_goal(cargoal)


    def exercise_problem(self, problem : unified_planning.model.Problem, expected_robustness_result : up.solvers.PlanGenerationResultStatus):
        w = PDDLWriter(problem)
        with open("kaka_domain.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("kaka_problem.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)
            self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

        grounder = Grounder(problem_kind=problem.kind)
        grounding_result = grounder.ground(problem)
        ground_problem = grounding_result.problem

        w = PDDLWriter(ground_problem)
        with open("kaka_domain_grounded.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("kaka_problem_grounded.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(ground_problem)
            self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

        unified_planning.model.agent.defineAgentsByFirstArg(ground_problem)

        #self.assertEqual(len(ground_problem.agents), 4)

        rv = RobustnessVerifier(ground_problem)

        rv_problem = rv.get_rewritten_problem()

        w = PDDLWriter(rv_problem)
        with open("kaka_domain_grounded_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("kaka_problem_grounded_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        with OneshotPlanner(problem_kind=rv_problem.kind) as planner:
            result = planner.solve(rv_problem)
            self.assertEqual(result.status, expected_robustness_result)


            
        

    def test_intersection_problem_pddl(self):
        reader = PDDLReader()
        
        domain_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'domain.pddl')
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'problem.pddl')
        problem = reader.parse_problem(domain_filename, problem_filename)

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

    def test_intersection_problem_lifted_pddl(self):
        reader = PDDLReader()
        
        domain_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'domain.pddl')
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'problem.pddl')
        problem = reader.parse_problem(domain_filename, problem_filename)

        at = problem.fluent("at")
        a1 = problem.object("a1")
        a2 = problem.object("a2")
        a3 = problem.object("a3")
        a4 = problem.object("a4")
        northex = problem.object("north-ex")
        southex = problem.object("south-ex")
        eastex = problem.object("east-ex")
        westex = problem.object("west-ex")
        
        ac1 = ExistingObjectAgent(a1, problem._env, [at(a1, northex)])
        ac2 = ExistingObjectAgent(a2, problem._env, [at(a2, southex)])
        ac3 = ExistingObjectAgent(a3, problem._env, [at(a3, eastex)])
        ac4 = ExistingObjectAgent(a4, problem._env, [at(a4, westex)])

        problem.add_agent(ac1)
        problem.add_agent(ac2)
        problem.add_agent(ac3)
        problem.add_agent(ac4)

        param_obj = problem.action("drive").parameter("a")
        agent_a = ExistingObjectAgent(param_obj)

        problem.action("drive").agent = agent_a
        problem.action("arrive").agent = agent_a

        rv = RobustnessVerifier(problem)

        rv_problem = rv.get_rewritten_problem()

        w = PDDLWriter(rv_problem)
        with open("kaka_domain_grounded_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("kaka_problem_grounded_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()


    def test_intersection_problem_interface_4cars(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north")
        self.add_car(problem, "c2", "north-ent", "south-ex", "south")
        self.add_car(problem, "c3", "west-ent", "east-ex", "east")
        self.add_car(problem, "c4", "east-ent", "west-ex", "west")

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

    def test_intersection_problem_interface_2cars_cross(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north")
        self.add_car(problem, "c3", "west-ent", "east-ex", "east")

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)        

    def test_intersection_problem_interface_2cars_opposite(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north")
        self.add_car(problem, "c2", "north-ent", "south-ex", "south")

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.UNSOLVABLE_PROVEN)        


