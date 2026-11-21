"""Finite evidence acquisition planning, independent of audit verdicts.

Ordinary uniform-cost search is deliberately used as the exact baseline too.
Domain specificity belongs to the versioned obligations and receipt validators,
not to this search algorithm. Advertised effects never constitute evidence.
"""
from dataclasses import dataclass
import hashlib
import heapq
import json
import math


@dataclass(frozen=True, order=True)
class EvidenceKey:
    kind: str
    identity: str
    version: str
    role: str


@dataclass(frozen=True)
class Acquisition:
    identifier: str
    requires: frozenset
    provides: frozenset
    cost: float

    def __post_init__(self):
        if not self.identifier or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError('action requires a name and a finite nonnegative cost')
        if not self.provides:
            raise ValueError('action has no advertised evidence')


@dataclass(frozen=True)
class EvidenceProblem:
    context: str
    present: frozenset
    goals: tuple  # (selected output name, frozenset of versioned obligations)
    actions: tuple
    blocked: tuple = ()

    def __post_init__(self):
        names=[a.identifier for a in self.actions]
        if len(set(names)) != len(names):raise ValueError('duplicate action identifiers')
        outputs=[name for name,_ in self.goals]
        if not outputs or len(set(outputs)) != len(outputs):raise ValueError('duplicate or empty outputs')

    def digest(self):
        key=lambda k:(k.kind,k.identity,k.version,k.role)
        payload={'context':self.context,'present':sorted(map(key,self.present)),
          'goals':[(name,sorted(map(key,items))) for name,items in self.goals],
          'actions':[(a.identifier,sorted(map(key,a.requires)),sorted(map(key,a.provides)),a.cost) for a in self.actions],
          'blocked':self.blocked}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EvidencePlan:
    problem_digest: str
    context: str
    actions: tuple
    declared_cost: float
    reachable: bool
    unresolved: frozenset
    expanded: int


def exact_plan(problem, *, max_states=100000):
    """Minimum declared cost for ALL declared goals, within a finite action set.

    If goals are unreachable return a plan for the maximum number of reachable
    obligation keys (ties by cost). This is not an optimum for audit decisions,
    acquisition outcomes, elapsed time, or whole-output coverage.
    """
    goals=frozenset(k for _,items in problem.goals for k in items)
    universe=problem.present|goals|frozenset(k for a in problem.actions for k in a.provides|a.requires)
    keys=sorted(universe);bit={k:1<<i for i,k in enumerate(keys)}
    mask=lambda items:sum(bit[k] for k in items)
    initial=mask(problem.present);wanted=mask(goals)
    actions=[(a,mask(a.requires),mask(a.provides)) for a in sorted(problem.actions,key=lambda a:a.identifier)]
    heap=[(0.,(),initial)];best={initial:(0.,())};best_partial=(initial& wanted).bit_count(),0.,(),initial
    expanded=0
    while heap:
        cost,path,state=heapq.heappop(heap)
        if best.get(state)!=(cost,path):continue
        expanded+=1
        if expanded>max_states:raise RuntimeError('finite exact search state limit exceeded; no optimality claim')
        rank=((state&wanted).bit_count(),-cost)
        if rank>(best_partial[0],best_partial[1]):best_partial=(rank[0],rank[1],path,state)
        if state&wanted==wanted:
            return EvidencePlan(problem.digest(),problem.context,path,cost,True,frozenset(),expanded)
        for a,req,effect in actions:
            if state&req!=req:continue
            after=state|effect
            if after==state:continue
            candidate=(cost+a.cost,path+(a.identifier,))
            if after not in best or candidate<best[after]:
                best[after]=candidate;heapq.heappush(heap,(*candidate,after))
    _,negcost,path,state=best_partial
    unresolved=frozenset(k for k in goals if not state&bit[k])
    return EvidencePlan(problem.digest(),problem.context,path,-negcost,False,unresolved,expanded)


def execute_plan(problem,plan,*,current_context,acquire,validate):
    """Callbacks are trusted adapters. No advertised effect is accepted directly.

    A failed or incomplete receipt ends this plan; callers may build a fresh
    problem with the failed action unavailable. No automatic policy or state
    modification, no training rerun, and no direct PASS assignment occurs here.
    """
    if plan.problem_digest!=problem.digest() or plan.context!=current_context():
        raise ValueError('evidence plan context changed')
    actions={a.identifier:a for a in problem.actions};facts=set(problem.present);receipts=[]
    for identifier in plan.actions:
        action=actions[identifier]
        if plan.context!=current_context():raise ValueError('evidence plan became stale during acquisition')
        if not action.requires<=facts:raise ValueError('evidence prerequisites not established')
        receipt=acquire(action)
        verified=frozenset(validate(action,receipt))
        if not verified<=action.provides:raise ValueError('validator returned undeclared effects')
        facts.update(verified)
        receipts.append({'action':identifier,'accepted':len(verified),'advertised':len(action.provides)})
        if verified!=action.provides:break
    goals=frozenset(k for _,items in problem.goals for k in items)
    return {'facts':frozenset(facts),'receipts':receipts,'complete':goals<=facts,
            'unresolved':goals-facts,'audit_decision_assigned':False}
