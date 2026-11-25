from src.dms.state_machine import DMSStateMachine


def _feed(machine, start, end, step, **kwargs):
    timestamp = start
    result = None
    while timestamp <= end:
        result = machine.update(timestamp=timestamp, **kwargs)
        timestamp += step
    return result


def test_normal_state_stays_normal():
    machine = DMSStateMachine()
    result = _feed(
        machine,
        0.0,
        1.0,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.30,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
    )
    assert result["state"] == DMSStateMachine.NORMAL


def test_sustained_head_turn_triggers_distraction():
    machine = DMSStateMachine(distraction_threshold=0.5)
    result = _feed(
        machine,
        0.0,
        0.8,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.30,
        mar=0.20,
        yaw=35.0,
        pitch=0.0,
    )
    assert result["state"] == DMSStateMachine.DISTRACTED


def test_closed_eyes_escalate_to_severe_drowsiness():
    machine = DMSStateMachine(
        drowsy_closed_threshold=0.3,
        severe_closed_threshold=0.7,
    )
    result = _feed(
        machine,
        0.0,
        0.9,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.10,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
    )
    assert result["state"] == DMSStateMachine.SEVERE_DROWSY


def test_lost_face_triggers_unavailable():
    machine = DMSStateMachine(unavailable_threshold=0.5)
    _feed(
        machine,
        0.0,
        0.1,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.30,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
    )
    result = _feed(
        machine,
        0.2,
        0.8,
        0.1,
        face_detected=False,
        quality_ok=False,
    )
    assert result["state"] == DMSStateMachine.UNAVAILABLE


def test_normal_recovers_after_risk_clears():
    machine = DMSStateMachine(
        distraction_threshold=0.3,
        recovery_threshold=0.2,
    )
    _feed(
        machine,
        0.0,
        0.5,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.30,
        mar=0.20,
        yaw=35.0,
        pitch=0.0,
    )
    result = _feed(
        machine,
        0.6,
        1.0,
        0.1,
        face_detected=True,
        quality_ok=True,
        ear=0.30,
        mar=0.20,
        yaw=0.0,
        pitch=0.0,
    )
    assert result["state"] == DMSStateMachine.NORMAL
