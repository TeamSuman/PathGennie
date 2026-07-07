"""Rule-based agentic controller tests."""

from pathgennie.agent import RuleBasedController, SwarmParams


def _controller(**kw):
    return RuleBasedController(SwarmParams(n_trial=8, tau1=4, tau2=8), **kw)


def test_escalates_when_stalling():
    c = _controller(stall_window=4, escalate=2.0)
    flat = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = c.update(flat)
    assert p.n_trial > 8          # swarm enlarged
    assert p.tau1 > 4             # sampler lengthened
    assert p.tau2 > 8


def test_relaxes_when_progressing():
    c = _controller(stall_window=4, relax=0.5)
    improving = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = c.update(improving)
    assert p.n_trial < 8          # swarm shrunk to save compute


def test_respects_bounds():
    c = _controller(n_bounds=(4, 10), tau1_bounds=(2, 5), tau2_bounds=(4, 9), escalate=10.0)
    p = c.update([0.0, 0.0, 0.0])
    assert p.n_trial <= 10 and p.tau1 <= 5 and p.tau2 <= 9


def test_should_stop_on_plateau():
    c = _controller(stop_patience=3, stall_eps=1e-6)
    # First call sets the best; subsequent flat calls accrue no-improvement count.
    for _ in range(5):
        c.update([1.0, 1.0, 1.0])
    assert c.should_stop([1.0, 1.0, 1.0])


def test_should_not_stop_while_improving():
    c = _controller(stop_patience=3)
    hist = [0.0]
    for v in range(1, 6):
        hist.append(float(v))
        c.update(hist)
    assert not c.should_stop(hist)


def test_refresh_schedule():
    c = _controller(refresh_every=10)
    assert c.should_refresh_cv(0) is True      # first time
    assert c.should_refresh_cv(5) is False
    assert c.should_refresh_cv(10) is True
    assert c.should_refresh_cv(11) is False


def test_choose_frontier_least_visited():
    assert RuleBasedController.choose_frontier([3, 1, 2, 5]) == 1
