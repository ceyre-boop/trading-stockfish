"""
Stress test event generator for the validation lab.
"""


def generate_stress_events():
    # Simulate high-frequency and edge-case events
    events = []
    for i in range(10000):
        events.append(
            {"timestamp": i, "type": "tick", "price": 100 + i % 10, "volume": 1000 + i}
        )
    # Add edge cases
    events.append({"timestamp": 10001, "type": "halt", "reason": "market halt"})
    events.append({"timestamp": 10002, "type": "resume", "reason": "market resume"})
    return events
