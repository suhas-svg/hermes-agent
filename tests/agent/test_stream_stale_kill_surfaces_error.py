"""A stale kill that cannot be recovered must SURFACE an error, not sit silent.

Regression for #132606. The chat-completions streaming path killed a stalled
attempt and then held the stall in silence: ``_kill_stale_stream`` never set
``result["error"]`` and never ran the stale breaker, so ``run()`` had nothing to
translate and the turn only ended when the stale watchdog expired a *second*
time. The in-repo Bedrock sibling (chat_completion_helpers.py:2757) already
surfaces a ``TimeoutError`` and runs the breaker; this asserts the
chat-completions path reaches the agent loop the same way.

The invariant is a RELATION between two pieces of state, not a snapshot: a kill
whose abort the worker recovered from must stay silent (the retry is in flight
and has to be allowed to land), while a kill that leaves the SAME stream attempt
current proves no retry was opened and must surface a retryable error.

Attempt bookkeeping is the real ``_StreamingCall`` state (``stream_attempt_lock``
/ ``stream_attempt_state``) rather than a stub, because whether the worker
advanced the attempt IS the decision under test.
"""

import threading

import pytest
from types import SimpleNamespace

from agent import chat_completion_helpers as h


def _agent():
    agent = SimpleNamespace(
        _buffer_diagnostic_status=lambda text: None,
        _emit_diagnostic_wait=lambda text: None,
        _touch_activity=lambda text: None,
    )
    agent._consecutive_stale_streams = 0
    return agent


def _stale_call(agent, *, attempt_id=1, worker=None):
    """A ``_StreamingCall`` with the REAL attempt bookkeeping and no HTTP."""
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = agent
    call.api_kwargs = {"model": "m"}
    call.last_chunk_time = {"t": 0.0}
    call.result = {"response": None, "error": None}
    call._stream_stale_timeout = 1.0
    call.stream_attempt_lock = threading.Lock()
    call.stream_attempt_state = {"current": attempt_id, "cancelled": set(),
                                 "discarded_chunks": 0, "discarded_bytes": 0}
    call._stale_counted_attempts = set()
    call._stale_killed_attempts = set()
    call._attempt_stream_response = None
    call.worker = worker
    return call


@pytest.fixture(autouse=True)
def _no_socket_io(monkeypatch):
    """The kill aborts a socket and cancels the attempt; keep both inert."""
    monkeypatch.setattr(h._StreamingCall, "_shutdown_stale_attempt_socket",
                        lambda self, response: None, raising=False)
    monkeypatch.setattr(h._StreamingCall, "_cancel_current_stream_attempt",
                        lambda self, reason: None, raising=False)
    monkeypatch.setattr(h._StreamingCall, "_close_managed_stream",
                        lambda self: None, raising=False)


def _alive_worker():
    """A worker thread object that reports alive (the wedged-transport case)."""
    worker = SimpleNamespace(is_alive=lambda: True)
    return worker


def _dead_worker():
    return SimpleNamespace(is_alive=lambda: False)


def test_recovered_kill_stays_silent_so_the_inflight_retry_can_land(monkeypatch):
    """A kill the worker RECOVERED from opens a new attempt. Surfacing an error
    there would strand a call that is about to succeed — #132606 is the wedge,
    not every kill."""
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _agent()
    call = _stale_call(agent, attempt_id=1, worker=_alive_worker())

    call._kill_stale_stream(2.0)

    assert call.result["error"] is None, "a recovered kill must leave the retry in flight"
    assert agent._consecutive_stale_streams == 1, "the breaker still advances per killed attempt"


def test_kill_repeated_on_the_same_attempt_surfaces_a_retryable_timeout(monkeypatch):
    """A DEAD event loop cannot unwind. The monitor resets ``last_chunk_time`` on
    every kill, so the abort gets a FULL stale window to unwind the reader; the same
    attempt id coming back is proof it never did. No retry is in flight, so the stall
    must reach the agent loop as a retryable TimeoutError instead of holding silent
    until the watchdog fires yet again (#132606)."""
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _agent()
    call = _stale_call(agent, attempt_id=1, worker=_alive_worker())

    call._kill_stale_stream(2.0)   # first kill: recovery attempt, stays silent
    assert call.result["error"] is None
    call._kill_stale_stream(4.0)   # same attempt survived a whole window

    assert isinstance(call.result["error"], TimeoutError)
    message = str(call.result["error"])
    assert "no chunks" in message
    assert "retry" in message.lower()
    # The breaker must see this attempt, or one dead loop burns 1 of 5 attempts
    # and the session limps for five turns instead of failing fast.
    assert agent._consecutive_stale_streams == 1


def test_kill_is_silent_once_the_worker_has_already_finished(monkeypatch):
    """A kill that lands after the worker exited on its own is not a wedge: the
    call already ended, and surfacing an error over a finished worker would
    clobber a real outcome."""
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _agent()
    call = _stale_call(agent, attempt_id=1, worker=_dead_worker())

    call._kill_stale_stream(2.0)
    call._kill_stale_stream(4.0)

    assert call.result["error"] is None, "a finished worker is not an unrecoverable stall"


def test_surfaced_timeout_stays_classified_retryable():
    """The surfaced error must enter the normal transient-retry path, never
    surface as a hard failure. A dead transport is transport-class, classified
    retryable at agent/error_classifier.py:943."""
    from agent.error_classifier import classify_api_error

    error = TimeoutError("OpenAI-compatible stream produced no chunks for 600s — dead stream; retrying.")
    classified = classify_api_error(error, provider="custom", model="m")
    assert classified.retryable, "a dead stream must stay retryable, not become a hard failure"


def test_tripped_breaker_surfaces_instead_of_killing_the_monitor_thread(monkeypatch):
    """``_check_stale_giveup`` RAISES past the threshold. It runs on the MONITOR
    thread, so an escaping exception would kill the loop that owns the call and
    leave ``run()`` with nothing to translate — the very silence this fixes. The
    give-up must arrive as ``result["error"]`` instead."""
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "1")
    agent = _agent()
    call = _stale_call(agent, attempt_id=1, worker=_alive_worker())

    call._kill_stale_stream(2.0)
    call._kill_stale_stream(4.0)

    assert isinstance(call.result["error"], RuntimeError), (
        "a tripped breaker must reach run() through result['error'], "
        "not by killing the monitor thread"
    )
    assert "unresponsive" in str(call.result["error"])
