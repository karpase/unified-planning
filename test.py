import unified_planning
from unified_planning.shortcuts import *
from unified_planning.io.pddl_writer import PDDLWriter
from unified_planning.io.pddl_reader import PDDLReader
import unified_planning.model.agent

# reader = PDDLReader()
# problem = reader.parse_problem('/home/karpase/git/unified-planning/unified_planning/test/pddl/depot/domain.pddl', '/home/karpase/git/unified-planning/unified_planning/test/pddl/depot/problem.pddl')

# print(problem)

# grounder = Grounder(problem_kind=problem.kind)
# grounding_result = grounder.ground(problem)
# ground_problem = grounding_result.problem
# print(ground_problem)

# print(len(ground_problem.actions))

# agents = unified_planning.model.agent.defineAgentsByFirstArg(ground_problem)
# for agent in agents:
#     print(agent.name, list(map(lambda a: a.name, agent.actions)), agent.goals)



loc = UserType("loc")
agent = UserType("agent")

at = Fluent('at', _signature=[Parameter('l', loc), Parameter('a', agent)])
free = Fluent('at', _signature=[Parameter('l', loc)])
start = Fluent('start', _signature=[Parameter('l', loc), Parameter('a', agent)])
arrived = Fluent('arrived', _signature=[Parameter('a', agent)])
crossed = Fluent('crossed', _signature=[Parameter('a', agent)])

problem = Problem('intersection')
problem.add_fluent(at)
problem.add_fluent(free)
problem.add_fluent(start)
problem.add_fluent(arrived)
problem.add_fluent(crossed)

print(problem)
