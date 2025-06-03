import os
import re
import json
import time
import pathlib

from letta_client import Letta
from letta_client.agents.client import MessageCreate
from tau_bench.envs import get_env

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT  = BASE / "letta_runs"
OUT.mkdir(exist_ok=True)

POLL = 0.5

def load_env(env_name: str, task_id: int):
    return get_env(
        env_name = env_name,
        user_strategy = "llm",
        user_model = "gpt-4o",
        task_split = "dev",
        user_provider = "openai",
        task_index = task_id,
    )

client = Letta(
    token    = os.environ["LETTA_API_KEY"],
    base_url = os.environ["LETTA_BASE_URL"],
)

def make_agent() -> str:
    agent = client.agents.create(
        model = "openai/gpt-4o-mini",
        embedding = "openai/text-embedding-3-small",
        tools = [],
        memory_blocks = [{
            "label": "persona",
            "value": (
                "You are an agent evaluated by TAU-Bench.\n"
                "When you need to act, output exactly:\n"
                "Action: tool_name(arg1=\"…\")\n"
                "and nothing else."
            ),
        }],
    )
    return agent.id

def send_and_wait(aid: str, user_text: str, poll: float = POLL) -> str:
    client.agents.messages.create(
        aid,
        messages=[MessageCreate(role="user", content=user_text)]
    )
    while True:
        msgs = client.agents.messages.list(aid, limit=1)
        if not msgs:
            time.sleep(poll)
            continue

        msg = msgs[0]

        if getattr(msg, "role", "") == "user":
            time.sleep(poll)
            continue

        if hasattr(msg, "content"):
            return msg.content

        time.sleep(poll)

def run_task(env_name: str, task_id: int):
    env, _ = load_env(env_name, task_id), None
    obs, _ = env.reset()                       
    aid  = make_agent()
    traj = []

    while True:
        user_text = obs[0]
        assistant = send_and_wait(aid, user_text)
        print(f"[{env_name} {task_id}] turn {len(traj)+1}")

        record = {"user": user_text, "assistant": assistant}
        if (m := re.search(r"^Action:\s*(.*)$", assistant, re.M)):
            record["logged_action"] = m.group(1)
        traj.append(record)

        obs, _reward, done, _info = env.step(assistant)
        if done:
            break

    out_path = OUT / f"{env_name}_{task_id}.json"
    out_path.write_text(json.dumps(traj, indent=2))
    print(f"✓ wrote {out_path.relative_to(BASE)}")

if __name__ == "__main__":
    for tid in (1, 2, 3):
        run_task("retail", tid)
