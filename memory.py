"""
memory.py — Lightweight persistent memory tree.
Stores user facts in a hierarchical JSON file. After each conversation,
the LLM extracts key facts which are merged into the tree.
No external database, no vector store — just fast JSON read/write.
"""

import json
import logging
import os
import threading
from typing import Optional

import config
import llm as llm_module

logger = logging.getLogger(__name__)


class MemoryTree:
    """
    Hierarchical JSON-based memory system.

    Structure:
    {
        "personal": {"name": "Alex", "age": "25"},
        "preferences": {"music": "jazz", "food": "sushi"},
        "interests": {"hobbies": ["coding", "gaming"]},
        "important_info": {"birthday": "March 15"}
    }
    """

    def __init__(self, filepath: str = config.MEMORY_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self.tree = self._load()
        logger.info(f"🧠 Memory loaded: {self._fact_count()} facts from {filepath}")

    def _load(self) -> dict:
        """Load memory tree from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️  Failed to load memory: {e}")
        return {}

    def _save(self):
        """Persist memory tree to disk."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.tree, f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 Memory saved: {self._fact_count()} facts")
        except IOError as e:
            logger.error(f"❌ Failed to save memory: {e}")

    def _fact_count(self) -> int:
        """Count total facts across all categories."""
        count = 0
        for category in self.tree.values():
            if isinstance(category, dict):
                count += len(category)
        return count

    def get_context_summary(self) -> str:
        """
        Build a concise text summary of memories for injection into the LLM prompt.
        Respects MEMORY_MAX_CONTEXT_LENGTH.
        """
        if not self.tree:
            return ""

        parts = []
        for category, facts in self.tree.items():
            if isinstance(facts, dict) and facts:
                fact_strs = []
                for key, value in facts.items():
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    fact_strs.append(f"  - {key}: {value}")
                parts.append(f"{category}:\n" + "\n".join(fact_strs))

        summary = "\n".join(parts)

        # Truncate if too long
        if len(summary) > config.MEMORY_MAX_CONTEXT_LENGTH:
            summary = summary[: config.MEMORY_MAX_CONTEXT_LENGTH] + "\n  ..."

        return summary

    def merge(self, new_facts: dict):
        """
        Merge new facts into the memory tree (additive).
        Existing keys are updated, new keys are added.
        """
        with self._lock:
            for category, facts in new_facts.items():
                if not isinstance(facts, dict):
                    continue

                if category not in self.tree:
                    self.tree[category] = {}

                for key, value in facts.items():
                    existing = self.tree[category].get(key)

                    # If both are lists, extend without duplicates
                    if isinstance(existing, list) and isinstance(value, list):
                        for item in value:
                            if item not in existing:
                                existing.append(item)
                    # If existing is a list and new is a string, append
                    elif isinstance(existing, list) and isinstance(value, str):
                        if value not in existing:
                            existing.append(value)
                    else:
                        self.tree[category][key] = value

            self._save()
            logger.info(f"🧠 Memory updated: {self._fact_count()} total facts")

    def extract_and_merge(self, user_text: str, assistant_text: str):
        """
        Ask the LLM to extract key facts from the conversation, then merge.
        Runs in a background thread to avoid blocking.
        """
        thread = threading.Thread(
            target=self._extract_worker,
            args=(user_text, assistant_text),
            daemon=True,
        )
        thread.start()

    def _extract_worker(self, user_text: str, assistant_text: str):
        """Background worker: extract facts from conversation via LLM."""
        try:
            conversation = (
                f"User said: {user_text}\n"
                f"Assistant said: {assistant_text}"
            )

            prompt = f"{config.MEMORY_EXTRACT_PROMPT}\n\nConversation:\n{conversation}"

            # Collect the full LLM response
            response = ""
            for token in llm_module.stream_response(prompt):
                response += token

            response = response.strip()

            # Try to parse JSON from the response
            # Handle cases where LLM wraps JSON in markdown code blocks
            if "```" in response:
                # Extract JSON from code block
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    response = response[start:end]

            new_facts = json.loads(response)

            if isinstance(new_facts, dict) and new_facts:
                self.merge(new_facts)
                logger.info(f"🧠 Extracted {len(new_facts)} fact categories from conversation")
            else:
                logger.debug("🧠 No new facts extracted from conversation.")

        except json.JSONDecodeError:
            logger.debug(f"🧠 Could not parse memory extraction response: {response[:100]}")
        except Exception as e:
            logger.warning(f"⚠️  Memory extraction failed: {e}")
