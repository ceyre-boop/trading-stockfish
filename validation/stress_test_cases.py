"""
Stress test event generator for the validation lab.
"""


def generate_stress_events():
    # Simulate high-frequency and edge-case events
    events = []
    for event_index in range(10000):
        events.append(
            {
                "timestamp": event_index,
                "type": "tick",
                "price": 100 + event_index % 10,
                "volume": 1000 + event_index,
            }
        )
    # Add edge cases
    events.append({"timestamp": 10001, "type": "halt", "reason": "market halt"})
    events.append({"timestamp": 10002, "type": "resume", "reason": "market resume"})
    return events
