"""
Phase 2 프롬프트 import 확인.
실행: 프로젝트 루트에서 python tests/test_prompts_import.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts.planner import PLANNER_SYSTEM, PLANNER_HUMAN
from prompts.generator import GENERATOR_SYSTEM, GENERATOR_HUMAN

print(PLANNER_SYSTEM[:100])
print(PLANNER_HUMAN[:100])
print(GENERATOR_SYSTEM[:100])
print(GENERATOR_HUMAN[:100])
print("OK")
