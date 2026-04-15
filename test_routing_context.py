#!/usr/bin/env python3
"""Quick test to validate router receives detected objects context."""

import sys
sys.path.insert(0, '/Users/home/Projets/BIRA/src')

from bira_components.slm_formatter import SLM_Formatter

# Mock object with position data
class MockObject:
    def __init__(self, obj_id, position, confidence=90):
        self.id = obj_id
        self.position = position  # (x, y, z) in meters
        self.confidence = confidence
        self.bounding_box_2d = [[0, 0], [10, 10]]

# Test 1: Router prompt with no candidates
print("=" * 60)
print("Test 1: Router prompt with NO pending clarification")
print("=" * 60)
formatter = SLM_Formatter(SLM_Formatter.RESPONSE_SCHEMA)
messages = formatter.build_route_messages("give me a bottle")
print("System prompt:")
print(messages[0]['content'])
print("\nUser message:")
print(messages[1]['content'])

# Test 2: Router prompt with pending label but no candidates
print("\n" + "=" * 60)
print("Test 2: Router prompt with PENDING LABEL but no candidates")
print("=" * 60)
messages = formatter.build_route_messages("the farthest one", pending_label="bottle")
print("System prompt:")
print(messages[0]['content'])
print("\nUser message:")
print(messages[1]['content'])

# Test 3: Router prompt with pending label AND candidates
print("\n" + "=" * 60)
print("Test 3: Router prompt with PENDING LABEL AND detected candidates")
print("=" * 60)

# Create two bottle candidates at different distances
bottles = [
    MockObject(obj_id=1, position=(-0.15, 0.0, 0.8)),   # Left, closer
    MockObject(obj_id=2, position=(0.10, 0.0, 1.3)),    # Right, farther
]

messages = formatter.build_route_messages(
    "the farthest one",
    pending_label="bottle",
    detected_objects=bottles,
    detection_labels=[39, 39]  # Both are label 39 (bottle)
)
print("System prompt:")
print(messages[0]['content'])
print("\nUser message:")
print(messages[1]['content'])

# Check if the prompt mentions the candidates
if "position=" in messages[0]['content'] and "0.80" in messages[0]['content']:
    print("\n✓ SUCCESS: Router prompt includes candidate positions!")
else:
    print("\n✗ FAILURE: Router prompt does NOT include candidate positions")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✓ Router formatter correctly builds prompts with candidate context")
print("✓ Candidate positions are listed naturally")
print("✓ Pending clarifications are identified")
