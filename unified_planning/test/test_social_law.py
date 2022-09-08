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
from unified_planning.transformers import InstantaneousActionRobustnessVerifier, DuativeActionRobustnessVerifier, NegativeConditionsRemover, SingleAgentProjection, SocialLaw, WaitingActionRobustnessVerifier


# (define (domain intersection)
 
#  (:requirements :strips :typing :multi-agent :unfactored-privacy)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DOMAINS_PATH = os.path.join(FILE_PATH, 'pddl')

class TestSocialLaws(TestCase):
    def setUp(self):
        TestCase.setUp(self)

    def create_basic_intersection_problem_interface(self, use_waiting : bool = False) -> unified_planning.model.Problem:
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
        #


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
        arrive.agent = ExistingObjectAgent(a)



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
        if use_waiting:
            drive.add_precondition_wait(free(l2))
        else:
            drive.add_precondition(free(l2))
        drive.add_precondition(traveldirection(a,d))
        drive.add_precondition(connected(l1,l2,d))
        #drive.add_precondition(yieldsto(l1,ly))
        #drive.add_precondition(free(ly))
        drive.add_effect(at(a,l2),True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(a,l1), False)
        drive.add_effect(free(l1), True)
        drive.agent = ExistingObjectAgent(a)

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


    def create_durative_intersection_problem_interface(self, use_waiting : bool = False) -> unified_planning.model.Problem:
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
        #


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
        arrive = DurativeAction('arrive', a=car, l=loc)
        arrive.set_fixed_duration(1)
        a = arrive.parameter('a')
        l = arrive.parameter('l')
        
        arrive.add_condition(StartTiming(),start(a, l))
        arrive.add_condition(StartTiming(),Not(arrived(a)))
        arrive.add_condition(OpenDurationInterval(StartTiming(), EndTiming()),free(l))
        arrive.add_effect(EndTiming(), at(a,l), True)
        arrive.add_effect(EndTiming(), free(l), False)
        arrive.add_effect(EndTiming(), arrived(a), True)
        arrive.agent = ExistingObjectAgent(a)



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
        drive = DurativeAction('drive', a=car, l1=loc, l2=loc, d=direction, ly=loc)
        drive.set_fixed_duration(1)
        a = drive.parameter('a')
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')
        ly = drive.parameter('ly')
        drive.add_condition(StartTiming(), at(a,l1))
        if use_waiting:
            drive.add_condition_wait(ClosedDurationInterval(StartTiming(), EndTiming()), free(l2))
        else:
            drive.add_condition(ClosedDurationInterval(StartTiming(), EndTiming()), free(l2))
        drive.add_condition(StartTiming(), traveldirection(a,d))
        drive.add_condition(EndTiming(), connected(l1,l2,d))
        #drive.add_precondition(yieldsto(l1,ly))
        #drive.add_precondition(free(ly))
        drive.add_effect(EndTiming(), at(a,l2),True)
        drive.add_effect(EndTiming(), free(l2), False)
        drive.add_effect(StartTiming(), at(a,l1), False)
        drive.add_effect(EndTiming(), free(l1), True)
        drive.agent = ExistingObjectAgent(a)

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


    def add_car(self, problem : unified_planning.model.Problem, name : str , startloc : str, endloc : str, cardirection : str, add_object : bool):
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

        caragent = ExistingObjectAgent(carobj, problem._env, [cargoal])
        problem.add_agent(caragent)

    def add_yields(self, problem : unified_planning.model.Problem, yields_list : List):
        loc = problem.user_type("loc")
        yieldsto = Fluent('yieldsto', _signature=[Parameter('l1', loc), Parameter('l2', loc)])
        problem.add_fluent(yieldsto, default_initial_value=False)

        free = problem.fluent("free")
        drive = problem.action("drive")
        l1 = drive.parameter('l1')
        ly = drive.parameter('ly')
        drive.add_precondition(yieldsto(l1,ly))
        drive.add_precondition_wait(free(ly))

        dummy_loc = unified_planning.model.Object("dummy", loc)
        problem.add_object(dummy_loc)

        yields = set()
        for l1_name, ly_name in yields_list:
            problem.set_initial_value(yieldsto(problem.object(l1_name), problem.object(ly_name)), True)                
            yields.add(problem.object(l1_name))
        for l1 in problem.objects(loc):
            if l1 not in yields:
                problem.set_initial_value(yieldsto(l1, dummy_loc), True)                
        




    def exercise_problem(self,                 
                problem : unified_planning.model.Problem, 
                expected_robustness_result : up.solvers.PlanGenerationResultStatus,
                perform_grounding : bool,
                infer_agents: bool,
                prefix : str):
        w = PDDLWriter(problem)
        with open(prefix + "_domain.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open(prefix + "_problem.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        with OneshotPlanner(name='fast_downward') as planner:
            result = planner.solve(problem)
            self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

        ref_problem = problem

        if perform_grounding:
            grounder = Grounder(name='tarski_grounder')
            grounding_result = grounder.ground(problem)
            ground_problem = grounding_result.problem

            ref_problem = ground_problem

            w = PDDLWriter(ground_problem)
            with open(prefix + "_domain_grounded.pddl","w") as f:
                print(w.get_domain(), file = f)
                f.close()
            with open(prefix + "_problem_grounded.pddl","w") as f:
                print(w.get_problem(), file = f)
                f.close()

            #with OneshotPlanner(problem_kind=problem.kind) as planner:
            #    result = planner.solve(ground_problem)
            #    self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

        if infer_agents:
            unified_planning.model.agent.defineAgentsByFirstArg(ref_problem)

        #self.assertEqual(len(ground_problem.agents), 4)

        for i, agent in enumerate(ref_problem.agents):
            sap = SingleAgentProjection(ref_problem, agent)
            sap_problem = sap.get_rewritten_problem()
            w = PDDLWriter(sap_problem)
            with open(prefix + "_domain_sap_" + str(i) + ".pddl","w") as f:
                print(w.get_domain(), file = f)
                f.close()
            with open(prefix + "_problem_sap_" + str(i) + ".pddl","w") as f:
                print(w.get_problem(), file = f)
                f.close()
            #with OneshotPlanner(problem_kind=sap_problem.kind) as planner:
            #    result = planner.solve(sap_problem)
            #    self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)

        # rv = InstantaneousActionRobustnessVerifier(problem)
        rv = WaitingActionRobustnessVerifier(problem)

        rv_problem = rv.get_rewritten_problem()

        w = PDDLWriter(rv_problem)
        with open(prefix + "_domain_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open(prefix + "_problem_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()


        ncr = NegativeConditionsRemover(rv_problem)
        ncr_rv_problem = ncr.get_rewritten_problem()

        w = PDDLWriter(ncr_rv_problem)
        with open(prefix + "_domain_ncr_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open(prefix + "_problem_ncr_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()        

        with OneshotPlanner(name='fast_downward') as planner: #problem_kind=ncr_rv_problem.kind) as planner:
            result = planner.solve(ncr_rv_problem)
            self.assertEqual(result.status, expected_robustness_result)


    def test_intersection_problem_pddl(self):
        reader = PDDLReader()

        domain_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'domain.pddl')
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, 'intersection', 'i1', 'problem.pddl')
        problem = reader.parse_problem(domain_filename, problem_filename)

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING, True, True, "pddl4cars")

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
        ac3 = ExistingObjectAgent(a3, problem._env, [at(a3, westex)])
        ac4 = ExistingObjectAgent(a4, problem._env, [at(a4, eastex)])

        problem.add_agent(ac1)
        problem.add_agent(ac2)
        problem.add_agent(ac3)
        problem.add_agent(ac4)

        param_obj = problem.action("drive").parameter("a")
        agent_a = ExistingObjectAgent(param_obj)

        problem.action("drive").agent = agent_a
        problem.action("arrive").agent = agent_a

        # rv = InstantaneousActionRobustnessVerifier(problem)
        rv = WaitingActionRobustnessVerifier(problem)

        rv_problem = rv.get_rewritten_problem()

        w = PDDLWriter(rv_problem)
        with open("pddll4cars_domain_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("pddll4cars_problem_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        ncr = NegativeConditionsRemover(rv_problem)
        ncr_rv_problem = ncr.get_rewritten_problem()

        w = PDDLWriter(ncr_rv_problem)
        with open("pddll4cars_domain_ncr_rv.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("pddll4cars_problem_ncr_rv.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()        

        with OneshotPlanner(name='fast_downward') as planner: #problem_kind=ncr_rv_problem.kind) as planner:
            result = planner.solve(ncr_rv_problem)
            self.assertEqual(result.status, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING)




    def test_intersection_problem_interface_4cars(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", False)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", False)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", False)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", False)

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING, True, True, "int4cars")

    def test_intersection_problem_interface_2cars_cross(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", False)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", False)

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.SOLVED_SATISFICING, True, True, "int2cars_cross")        

    def test_intersection_problem_interface_2cars_opposite(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", False)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", False)

        self.exercise_problem(problem, up.solvers.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY, True, True, "int2cars_opp")

    def test_intersection_problem_interface_lifted_4cars(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)

        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL)

    def test_intersection_problem_interface_lifted_2cars_cross(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)

        self.add_yields(problem, [])

        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL)

    def test_intersection_problem_interface_lifted_2cars_opposite(self):
        problem = self.create_basic_intersection_problem_interface()

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)

        self.add_yields(problem, [])

        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.ROBUST_RATIONAL)


    def test_intersection_problem_interface_lifted_4cars_deadlock(self):
        problem = self.create_basic_intersection_problem_interface(use_waiting=True)

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)

        self.add_yields(problem, [])
    
        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)

    def test_intersection_problem_interface_lifted_4cars_yield_deadlock(self):
        problem = self.create_basic_intersection_problem_interface(use_waiting=True)

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)

        self.add_yields(problem, [("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")])
        
        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)


    def test_intersection_problem_interface_lifted_4cars_yield_deadlock(self):
        problem = self.create_basic_intersection_problem_interface(use_waiting=True)

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)

        self.add_yields(problem, [("south-ent", "east-ent"),("north-ent", "west-ent")])
        
        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)


    def test_intersection_problem_interface_lifted_4cars_yield_robust(self):
        problem = self.create_basic_intersection_problem_interface(use_waiting=True)

        self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
        self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
        self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
        self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)

        self.add_yields(problem, [("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")])
        
        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(problem, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.ROBUST_RATIONAL)

    def test_counterexample(self):
        p = Problem("counterexample")

        f1 = Fluent("f1")
        nf1 = Fluent("nf1")
        f2 = Fluent("f2")
        nf2 = Fluent("nf2")
        f3 = Fluent("f3")
        nf3 = Fluent("nf3")
        p.add_fluent(f1, default_initial_value=True)
        p.add_fluent(nf1, default_initial_value=False)
        p.add_fluent(f2, default_initial_value=False)
        p.add_fluent(nf2, default_initial_value=True)
        p.add_fluent(f3, default_initial_value=True)
        p.add_fluent(nf3, default_initial_value=False)

        ag1 = Agent("ag1", p.env, [], "agent")
        ag1.add_goal(nf1)
        ag2 = Agent("ag2", p.env, [], "agent")
        ag2.add_goal(f2)
        p.add_agent(ag1)
        p.add_agent(ag2)

        a11 = InstantaneousAction("a11")
        a11.agent = ag1
        a11.add_precondition_wait(f3)
        a11.add_effect(f3, True)
        p.add_action(a11)

        a12 = InstantaneousAction("a12")
        a12.agent = ag1
        a12.add_effect(f1, False)
        a12.add_effect(f3, False)
        a12.add_effect(nf1, True)
        a12.add_effect(nf3, True)
        p.add_action(a12)

        a21 = InstantaneousAction("a21")
        a21.agent = ag2
        a21.add_effect(f2, True)
        a21.add_effect(f3, True)
        a21.add_effect(nf2, False)
        a21.add_effect(nf3, False)
        p.add_action(a21)

        a22 = InstantaneousAction("a22")
        a22.agent = ag2
        a22.add_precondition(nf2)
        a22.add_effect(f3, False)
        a22.add_effect(nf3, True)
        p.add_action(a22)

        w = PDDLWriter(p)
        with open("ce_domain.pddl","w") as f:
            print(w.get_domain(), file = f)
            f.close()
        with open("ce_problem.pddl","w") as f:
            print(w.get_problem(), file = f)
            f.close()

        planner = OneshotPlanner(name='fast_downward')
        l = SocialLaw(p, planner)
        status = l.is_robust()

        self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.ROBUST_RATIONAL)



    # def test_intersection_problem_durative(self):
    #     problem = self.create_durative_intersection_problem_interface(use_waiting=True)
    #
    #     self.add_car(problem, "c1", "south-ent", "north-ex", "north", True)
    #     self.add_car(problem, "c2", "north-ent", "south-ex", "south", True)
    #     self.add_car(problem, "c3", "west-ent", "east-ex", "east", True)
    #     self.add_car(problem, "c4", "east-ent", "west-ex", "west", True)
    #
    #     #self.add_yields(problem, [("south-ent", "east-ent"),("north-ent", "west-ent")])
    #
    #     #planner = OneshotPlanner(name='fast_downward')
    #     #l = SocialLaw(problem, planner)
    #     #status = l.is_robust()
    #
    #     #self.assertEqual(status, up.transformers.social_law.SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)
    #
    #     rv = DuativeActionRobustnessVerifier(problem, compile_away_numeric=False)
    #     rv_problem = rv.get_rewritten_problem()
    #
    #     w = PDDLWriter(rv_problem)
    #     with open("d_domain.pddl","w") as f:
    #         print(w.get_domain(), file = f)
    #         f.close()
    #     with open("d_problem.pddl","w") as f:
    #         print(w.get_problem(), file = f)
    #         f.close()
    #
    #     planner = OneshotPlanner(name='tamer')
    #     result = planner.solve(rv_problem)
        



