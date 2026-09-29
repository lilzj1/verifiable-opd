"""Explicit domain response budgets; no answer-based routing or filtering."""


def sampling_for_domain(sampling_params, domain, limits, tensor_response_limit):
    params = dict(sampling_params)
    if not limits:
        return params
    if domain not in limits:
        raise ValueError(f"Unknown text domain: {domain!r}")
    limit = limits[domain]
    if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= tensor_response_limit:
        raise ValueError("Domain generation budget must fit the allocated response tensor")
    params.pop("max_new_tokens", None)
    params["max_tokens"] = limit
    return params
