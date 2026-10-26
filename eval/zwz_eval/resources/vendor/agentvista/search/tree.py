"""Search node and tree for single-path (greedy) reasoning state."""

import copy
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class SearchNode:
    """State at one step: conversation history, images, turn count, final answer."""

    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    image_map: Dict[str, Image.Image] = field(default_factory=dict)
    current_turn: int = 0
    parent: Optional["SearchNode"] = None
    children: List["SearchNode"] = field(default_factory=list)
    score: float = 0.0
    evaluation_analysis: str = ""
    is_final: bool = False
    final_answer: Optional[str] = None
    node_id: str = ""
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    api_conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_token_count: int = 0
    pending_step_experiences: List[str] = field(default_factory=list)
    realtime_experiences_history: List[List[str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"node_{int(time.time() * 1000000)}"

    def copy(self) -> "SearchNode":
        n = SearchNode(
            conversation_history=copy.deepcopy(self.conversation_history),
            image_map=copy.deepcopy(self.image_map),
            current_turn=self.current_turn,
            score=self.score,
            evaluation_analysis=self.evaluation_analysis,
            is_final=self.is_final,
            final_answer=self.final_answer,
            node_id=f"{self.node_id}_copy_{int(time.time() * 1000000)}",
            depth=self.depth,
            api_conversation_history=copy.deepcopy(self.api_conversation_history),
            current_token_count=self.current_token_count,
            pending_step_experiences=copy.deepcopy(self.pending_step_experiences),
            realtime_experiences_history=copy.deepcopy(self.realtime_experiences_history),
        )
        for attr in ["turn_offset", "observations", "image_size_used_list", "save_dir_override"]:
            if hasattr(self, attr):
                try:
                    v = getattr(self, attr)
                    setattr(n, attr, v.copy() if isinstance(v, list) else v)
                except Exception:
                    pass
        return n

    def add_child(self, child: "SearchNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def get_path_to_root(self) -> List["SearchNode"]:
        path = []
        cur = self
        while cur:
            path.append(cur)
            cur = cur.parent
        return list(reversed(path))

    def update_score(self, score: float, analysis: str = "") -> None:
        self.score = score
        self.evaluation_analysis = analysis

    def mark_final(self, answer: str) -> None:
        self.is_final = True
        self.final_answer = answer

    def to_trajectory_text(self) -> str:
        out = ""
        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts) or "[Image]"
            out += f"**{role}**: {content}\n\n"
        if self.final_answer:
            out += f"**Final Answer**: {self.final_answer}\n"
        return out

    def estimate_tokens(self) -> int:
        total = 0
        for msg in self.conversation_history:
            c = msg.get("content", "")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        total += len(p.get("text", ""))
        return total // 4

    def can_expand(self, max_turns: int, max_images: int, max_tokens: int) -> bool:
        if self.is_final or self.current_turn >= max_turns or len(self.image_map) >= max_images:
            return False
        if self.estimate_tokens() >= max_tokens:
            return False
        return True


class SearchTree:
    """Container for root node (used by search strategies; greedy uses single path)."""

    def __init__(self, root: SearchNode):
        self.root = root
