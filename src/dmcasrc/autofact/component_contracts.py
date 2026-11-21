"""Limited construction contracts for sklearn 1.8.0, development revision.

An observed parent fit creates fresh predictive children whose fitting members
are bounded by that fit's members. Existing/replaced children are never adopted.
The bound only discharges a gap when disjoint from assessment; it never proves
an exposure. This certificate is valid only within the creating parent's output.
It is not a standalone tree provenance record or a restoration contract.
"""

CONTRACT_VERSION='owned-construction-v1/sklearn-1.8.0'

def ownership_contract(value):
    from .component_registry import component_spec
    spec = component_spec(value)
    # Keep the public version override used by contract-invalidation tests.
    if spec is None:return None
    return spec.identifier.replace('owned-construction-v1/sklearn-1.8.0', CONTRACT_VERSION)

def owned_descendants(parent):
    from .state_lineage import predictive_components
    seen={id(parent)};result=[]
    def visit(value):
        for _,child in predictive_components(value):
            if id(child) in seen:continue
            seen.add(id(child));result.append(child);visit(child)
    visit(parent)
    return result
