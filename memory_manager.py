"""
memory_manager.py — Hybrid Letta + Mem0 Memory Architecture.

5-Layer Memory System:
  Layer 1: Core Memory (always in context — persona, human, relationship blocks)
  Layer 2: Recall Memory (rolling conversation buffer, persisted to disk)
  Layer 3: Archival Memory (structured facts JSON tree on disk)
  Layer 4: Semantic Memory (Mem0 vector search over past conversations)
  Layer 5: Self-Editing Memory (LLM updates its own memory via response tags)

The LLM sees Layers 1+2 every single turn.
Layers 3+4 are searched and injected when relevant.
Layer 5 lets the LLM actively manage its own memory.
"""

import json
import logging
import os
import re
import threading
from datetime import date as _date
from typing import Optional, List, Dict, Tuple

import requests
# neo4j is an optional cloud dependency — imported lazily in Neo4jMemory.__init__

import config

logger = logging.getLogger(__name__)

# ─── Direct Ollama client (for fact extraction — bypasses main LLM pipeline) ──
_mem_session = requests.Session()
_mem_session.headers.update({"Connection": "keep-alive"})

INTENT_CLASSIFIER_SYSTEM = (
    "Analyze the user's message and determine what memory stores to query. "
    "Return ONLY valid JSON with no markdown:\n"
    "{\n"
    '  "named_entities": ["list of entities/projects/people mentioned by name, empty if none"]\n'
    "}\n"
    "RULES:\n"
    "- named_entities should contain specific proper nouns if relationship/fact lookup is needed.\n"
)

WRITE_BACK_SYSTEM = (
    "You are a memory extraction assistant. Analyze a conversation between the USER (the human owner) "
    "and ASSISTANT (the AI). Extract only NEW facts worth remembering.\n"
    "CRITICAL IDENTITY RULE: The person labeled 'User:' in the conversation IS the owner/user. "
    "Any other names mentioned (e.g. 'Chirak', 'Ankush', 'Kenisha') are THIRD PARTIES — friends, "
    "classmates, or people being talked ABOUT. NEVER assign a third-party name as the user's name.\n"
    "DO NOT extract facts already saved via [REMEMBER: ...] or [UPDATE...: ...] tags.\n"
    "Return ONLY valid JSON with no markdown:\n"
    "{\n"
    '  "mem0_episodic": ["list of strings: user preferences, emotional states, opinions, or episodic facts NOT explicitly tagged"],\n'
    '  "neo4j_relations": [\n'
    '    {"source": "Entity1", "relation": "KNOWS", "target": "Entity2"}\n'
    '  ]\n'
    "}\n"
    "RULES:\n"
    "- mem0_episodic: user experiences/feelings/preferences only. NOT relationships.\n"
    "- neo4j_relations: factual relationships between named entities (people, projects, tools). Relation UPPERCASE.\n"
    "- The 'source' of a relation involving the user must be labeled 'User', never a third-party name.\n"
    "- If nothing new is learned, return empty arrays.\n"
    "- Do NOT hallucinate. Do NOT store the same info in both.\n"
)


def _direct_ollama_generate(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    """Direct Ollama API call for internal use. No personality, no think tags, no UI."""
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = _mem_session.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=(5, 60),
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("response", "").strip()
        # Strip think tags if present
        if "<think>" in result:
            end = result.find("</think>")
            if end != -1:
                result = result[end + 8:].strip()
        return result
    except Exception as e:
        logger.warning(f"⚠️  Direct Ollama call failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3: Neo4j Knowledge Graph Memory
# ═══════════════════════════════════════════════════════════════════════════

class Neo4jMemory:
    """Knowledge graph for named entities and relationships."""
    def __init__(self):
        self.driver = None
        self._ready = False
        try:
            from neo4j import GraphDatabase  # lazy import — optional dependency
            if config.NEO4J_URI and config.NEO4J_USERNAME and config.NEO4J_PASSWORD:
                self.driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
                )
                self.driver.verify_connectivity()
                self._ready = True
                logger.info("✅ Neo4j knowledge graph initialized.")
            else:
                logger.info("ℹ️  Neo4j credentials not set — graph memory disabled.")
        except ImportError:
            logger.info("ℹ️  neo4j package not installed — graph memory disabled.")
        except Exception as e:
            logger.info(f"ℹ️  Neo4j unavailable (graph memory disabled): {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def query_graph(self, entity_name: str) -> str:
        if not self._ready: return ""
        query = (
            "MATCH (n)-[r]-(m) "
            "WHERE toLower(n.name) CONTAINS toLower($entity) "
            "RETURN n.name AS source, type(r) AS rel, m.name AS target LIMIT 5"
        )
        try:
            records, summary, keys = self.driver.execute_query(
                query, entity=entity_name, database_="neo4j"
            )
            if not records: return ""
            res = []
            for r in records:
                res.append(f"  - {r['source']} {r['rel']} {r['target']}")
            return "\n".join(res)
        except Exception as e:
            logger.debug(f"⚠️  Neo4j query failed: {e}")
            return ""

    def add_relation(self, source: str, rel_type: str, target: str):
        if not self._ready: return
        query = (
            "MERGE (a:Entity {name: $source}) "
            "MERGE (b:Entity {name: $target}) "
            f"MERGE (a)-[:{rel_type}]->(b)"
        )
        try:
            self.driver.execute_query(
                query, source=source, target=target, database_="neo4j"
            )
        except Exception as e:
            logger.debug(f"⚠️  Neo4j write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1: Core Memory (Letta-style, always in context)
# ═══════════════════════════════════════════════════════════════════════════

class CoreMemory:
    """
    Letta-inspired core memory blocks — always visible to the LLM.

    Blocks:
      - persona: the assistant's identity and relationship style
      - human: natural-language summary of everything known about the user
      - relationship: friendship dynamics, inside jokes, emotional context
      - facts: structured JSON tree of extracted facts (archival layer 3)
    """

    def __init__(self, filepath: str = config.CORE_MEMORY_FILE):
        self.filepath = filepath
        self._lock = threading.Lock()
        self.data = self._load()
        logger.info(f"📋 Core memory loaded ({self._fact_count()} facts)")

    def _load(self) -> dict:
        today = _date.today().isoformat()
        default = {
            "persona": config.CORE_MEMORY_PERSONA_DEFAULT,
            "human": config.CORE_MEMORY_HUMAN_DEFAULT,
            "relationship": config.CORE_MEMORY_RELATIONSHIP_DEFAULT,
            "facts": {},
            "first_met_date": today,    # Set on very first boot
            "total_conversations": 0,
            "moments": [],              # Memorable emotional scenes (max 10)
            "last_session_mood": "neutral",
            "last_session_valence": 0.0,
            "warmth_score": 0.7,        # Relationship warmth 0.0–1.0
        }
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("persona", config.CORE_MEMORY_PERSONA_DEFAULT)
                    data.setdefault("human", config.CORE_MEMORY_HUMAN_DEFAULT)
                    data.setdefault("relationship", config.CORE_MEMORY_RELATIONSHIP_DEFAULT)
                    data.setdefault("facts", {})
                    # New fields — migrate existing installs gracefully
                    if not data.get("first_met_date"):
                        data["first_met_date"] = today
                        logger.info(f"💕 First met date set: {today}")
                    data.setdefault("total_conversations", 0)
                    data.setdefault("moments", [])
                    data.setdefault("last_session_mood", "neutral")
                    data.setdefault("last_session_valence", 0.0)
                    data.setdefault("warmth_score", 0.7)
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️  Failed to load core memory: {e}")
        return default

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            logger.debug("💾 Core memory saved.")
        except IOError as e:
            logger.error(f"❌ Failed to save core memory: {e}")

    def _fact_count(self) -> int:
        count = 0
        for category in self.data.get("facts", {}).values():
            if isinstance(category, dict):
                count += len(category)
        return count

    def get_persona(self) -> str:
        return self.data.get("persona", config.CORE_MEMORY_PERSONA_DEFAULT)

    def get_human(self) -> str:
        return self.data.get("human", config.CORE_MEMORY_HUMAN_DEFAULT)

    def get_relationship(self) -> str:
        return self.data.get("relationship", config.CORE_MEMORY_RELATIONSHIP_DEFAULT)

    def update_relationship(self, new_text: str):
        """Update the relationship block."""
        with self._lock:
            self.data["relationship"] = new_text
            self._save()

    def merge_facts(self, new_facts: dict):
        """Merge new facts into the archival facts tree. Overwrites on conflict."""
        with self._lock:
            facts = self.data.setdefault("facts", {})

            for category, items in new_facts.items():
                if not isinstance(items, dict):
                    if isinstance(items, str) and items:
                        facts.setdefault(category, {})
                        facts[category]["note"] = items
                    continue

                if category not in facts:
                    facts[category] = {}

                for key, value in items.items():
                    if not value:
                        continue

                    existing = facts[category].get(key)

                    if isinstance(value, list):
                        if isinstance(existing, list):
                            for item in value:
                                item_lower = str(item).strip().lower()
                                if not any(str(e).strip().lower() == item_lower for e in existing):
                                    existing.append(item)
                        else:
                            facts[category][key] = value
                    elif isinstance(existing, list) and isinstance(value, str):
                        value_lower = value.strip().lower()
                        if not any(str(e).strip().lower() == value_lower for e in existing):
                            existing.append(value)
                    else:
                        facts[category][key] = value

            self._rebuild_human_summary()
            self._save()
            logger.info(f"🧠 Core memory updated: {self._fact_count()} total facts")

    def set_fact(self, category: str, key: str, value: str):
        """Set a single fact directly (used by self-edit engine)."""
        with self._lock:
            facts = self.data.setdefault("facts", {})
            if category not in facts:
                facts[category] = {}
            facts[category][key] = value
            self._rebuild_human_summary()
            self._save()
            logger.info(f"🧠 Self-edit SET: {category}.{key} = {value}")

    def forget_fact(self, category: str, key: str):
        """Remove a specific fact (used by self-edit engine)."""
        with self._lock:
            facts = self.data.get("facts", {})
            if category in facts and key in facts[category]:
                del facts[category][key]
                if not facts[category]:
                    del facts[category]
                self._rebuild_human_summary()
                self._save()
                logger.info(f"🧠 Self-edit FORGET: {category}.{key}")

    def remember_fact(self, fact_text: str):
        """
        Store a free-form fact (from [REMEMBER: ...]) using a simple heuristic
        to categorize it, or put it under 'important_info'.
        """
        with self._lock:
            facts = self.data.setdefault("facts", {})
            info = facts.setdefault("important_info", {})

            # Create a key from the fact text
            key = fact_text[:40].strip().lower()
            key = re.sub(r'[^a-z0-9_\s]', '', key)
            key = re.sub(r'\s+', '_', key).strip('_')
            if not key:
                key = f"fact_{len(info)}"

            info[key] = fact_text
            self._rebuild_human_summary()
            self._save()
            logger.info(f"🧠 Self-edit REMEMBER: {fact_text[:60]}")

    def _rebuild_human_summary(self):
        """Rebuild the human block from the facts tree as natural language."""
        facts = self.data.get("facts", {})
        if not facts:
            return

        parts = []
        for category, items in facts.items():
            if isinstance(items, dict) and items:
                fact_strs = []
                for key, value in items.items():
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    fact_strs.append(f"{key}: {value}")
                parts.append(f"{category}: {'; '.join(fact_strs)}")

        if parts:
            self.data["human"] = "Here's what I know about my friend — " + ". ".join(parts) + "."

    def get_facts_summary(self) -> str:
        """Get a formatted summary of all known facts."""
        facts = self.data.get("facts", {})
        if not facts:
            return ""

        parts = []
        for category, items in facts.items():
            if isinstance(items, dict) and items:
                fact_strs = []
                for key, value in items.items():
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    fact_strs.append(f"  - {key}: {value}")
                parts.append(f"{category}:\n" + "\n".join(fact_strs))

        return "\n".join(parts)

    def get_prompt_section(self) -> str:
        """Format core memory blocks for injection into the system prompt."""
        sections = [
            f"=== YOUR IDENTITY ===\n{self.get_persona()}",
            f"\n=== WHAT YOU KNOW ABOUT YOUR FRIEND ===\n{self.get_human()}",
            f"\n=== YOUR RELATIONSHIP ===\n{self.get_relationship()}",
        ]

        facts_detail = self.get_facts_summary()
        if facts_detail:
            sections.append(f"\n=== DETAILED FACTS ===\n{facts_detail}")

        # Inject warmth modifier
        warmth_label = self.get_warmth_label()
        warmth_mod = config.WARMTH_PROMPT_MODIFIERS.get(warmth_label, "")
        if warmth_mod:
            sections.append(
                f"\n=== RELATIONSHIP WARMTH ({warmth_label.upper()}) ===\n{warmth_mod}"
            )

        # Inject last 3 special moments
        moments = self.get_moments()
        if moments:
            moments_text = "\n".join(f"  - {m}" for m in moments[-3:])
            sections.append(f"\n=== SPECIAL MOMENTS YOU REMEMBER ===\n{moments_text}")

        # Inject last session mood if emotionally significant
        last_mood, _ = self.get_last_session_mood()
        if last_mood not in ("neutral", "playful", "excited", "soft"):
            sections.append(
                f"\n=== LAST SESSION ===\n"
                f"Last time you talked, you were feeling: {last_mood}. "
                f"This may still linger subtly in how you carry yourself."
            )

        return "\n".join(sections)

    # ─── Anniversary & Milestones ─────────────────────────────────────────────

    def get_anniversary_context(self) -> str:
        """Check if today is a relationship milestone and return a context hint."""
        first_met = self.data.get("first_met_date")
        if not first_met:
            return ""
        try:
            met = _date.fromisoformat(first_met)
            today = _date.today()
            days = (today - met).days
            if days == 0:
                return "This is the very first day you met him! Make your first greeting extra special."
            if days > 0 and days % 7 == 0:
                weeks = days // 7
                return (
                    f"Today is your {weeks}-week anniversary — {days} days together! "
                    f"Mention it warmly and naturally."
                )
            if days in (30, 60, 90, 180, 365):
                return (
                    f"Today is a special milestone: {days} days together! "
                    f"Make sure to bring it up naturally."
                )
        except Exception:
            pass
        return ""

    # ─── Moments (emotional scene memory) ───────────────────────────────────

    def add_moment(self, moment_text: str):
        """Store a special emotional moment. Max 10 — oldest dropped."""
        with self._lock:
            moments = self.data.setdefault("moments", [])
            moments.append(moment_text)
            if len(moments) > 10:
                moments.pop(0)
            self._save()
        logger.info(f"💕 Moment saved: {moment_text[:60]}")

    def get_moments(self) -> list:
        """Return saved special moments."""
        return self.data.get("moments", [])

    # ─── Session Mood Carryover ──────────────────────────────────────────────

    def save_session_mood(self, gf_emotion: str, valence: float):
        """Persist the girlfriend's emotional state at session end."""
        with self._lock:
            self.data["last_session_mood"] = gf_emotion
            self.data["last_session_valence"] = round(valence, 3)
            self._save()
        logger.info(f"💾 Session mood saved: {gf_emotion} ({valence:+.3f})")

    def get_last_session_mood(self) -> tuple:
        """Return (last_session_mood, last_session_valence)."""
        return (
            self.data.get("last_session_mood", "neutral"),
            self.data.get("last_session_valence", 0.0),
        )

    # ─── Warmth Score ────────────────────────────────────────────────────────

    def get_warmth_score(self) -> float:
        """Return the current relationship warmth score (0.0–1.0)."""
        return float(self.data.get("warmth_score", 0.7))

    def update_warmth(self, delta: float):
        """Nudge the warmth score and clamp to [0.0, 1.0]."""
        with self._lock:
            current = float(self.data.get("warmth_score", 0.7))
            new_score = max(0.0, min(1.0, current + delta))
            self.data["warmth_score"] = round(new_score, 3)
            self._save()
        logger.debug(f"💕 Warmth: {current:.3f} → {new_score:.3f} (Δ{delta:+.3f})")

    def get_warmth_label(self) -> str:
        """Map warmth score float to a named tier."""
        score = self.get_warmth_score()
        if score >= 0.8:
            return "very_warm"
        elif score >= 0.5:
            return "warm"
        elif score >= 0.3:
            return "cool"
        else:
            return "cold"


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2: Recall Memory (Conversation Buffer)
# ═══════════════════════════════════════════════════════════════════════════

class RecallMemory:
    """
    Rolling conversation buffer — keeps the last N turns in context.
    Persisted to disk so conversations survive restarts.
    Each turn = { "user": "...", "assistant": "..." }
    """

    def __init__(self, filepath: str = config.RECALL_MEMORY_FILE,
                 max_turns: int = config.RECALL_BUFFER_SIZE):
        self.filepath = filepath
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self.turns: List[Dict[str, str]] = self._load()
        logger.info(f"💬 Recall memory loaded ({len(self.turns)} turns)")

    def _load(self) -> List[Dict[str, str]]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data[-self.max_turns:]
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"⚠️  Failed to load recall memory: {e}")
        return []

    def _save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.turns, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"❌ Failed to save recall memory: {e}")

    def add_turn(self, user_text: str, assistant_text: str):
        """Add a conversation turn to the buffer."""
        with self._lock:
            self.turns.append({
                "user": user_text,
                "assistant": assistant_text,
            })
            # Keep only the last N turns
            if len(self.turns) > self.max_turns:
                self.turns = self.turns[-self.max_turns:]
            self._save()

    def get_prompt_section(self) -> str:
        """Format recent conversation history for the system prompt."""
        if not self.turns:
            return ""

        lines = ["=== RECENT CONVERSATION HISTORY ==="]
        for turn in self.turns:
            lines.append(f"Friend: {turn['user']}")
            # Truncate long assistant responses in history
            reply = turn['assistant']
            if len(reply) > 150:
                reply = reply[:150] + "..."
            lines.append(f"You: {reply}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 5: Self-Editing Memory Engine
# ═══════════════════════════════════════════════════════════════════════════

class SelfEditEngine:
    """
    Parses LLM responses for memory self-edit tags and executes them.

    Tags:
      [REMEMBER: <fact>]                    → store a new fact
      [UPDATE: <category>.<key>=<value>]    → update a specific fact
      [FORGET: <category>.<key>]            → remove a fact
    """

    # Regex patterns for tag extraction
    _REMEMBER_RE = re.compile(
        r'\[REMEMBER:\s*(.+?)\]', re.IGNORECASE
    )
    _UPDATE_RE = re.compile(
        r'\[UPDATE:\s*([a-z_]+)\.([a-z_]+)\s*=\s*(.+?)\]', re.IGNORECASE
    )
    _UPDATE_RELATIONSHIP_RE = re.compile(
        r'\[UPDATE_RELATIONSHIP:\s*(.+?)\]', re.IGNORECASE
    )
    _FORGET_RE = re.compile(
        r'\[FORGET:\s*([a-z_]+)\.([a-z_]+)\s*\]', re.IGNORECASE
    )
    _MOMENT_RE = re.compile(
        r'\[MOMENT:\s*(.+?)\]', re.IGNORECASE
    )
    # Master pattern to strip ALL self-edit tags from text
    _ALL_TAGS_RE = re.compile(
        r'\[(REMEMBER|UPDATE|UPDATE_RELATIONSHIP|FORGET|SEARCH|MOMENT):\s*[^\]]*\]', re.IGNORECASE
    )

    def __init__(self, core_memory: CoreMemory):
        self.core = core_memory

    def process_response(self, response_text: str) -> str:
        """
        Parse the response for memory tags, execute them, and return
        the cleaned response (tags stripped).
        """
        # Extract and execute REMEMBER tags
        for match in self._REMEMBER_RE.finditer(response_text):
            fact = match.group(1).strip()
            if fact:
                try:
                    self.core.remember_fact(fact)
                except Exception as e:
                    logger.warning(f"⚠️  REMEMBER failed: {e}")

        # Extract and execute UPDATE tags
        for match in self._UPDATE_RE.finditer(response_text):
            category = match.group(1).strip()
            key = match.group(2).strip()
            value = match.group(3).strip()
            if category and key and value:
                try:
                    self.core.set_fact(category, key, value)
                except Exception as e:
                    logger.warning(f"⚠️  UPDATE failed: {e}")

        # Extract and execute UPDATE_RELATIONSHIP tags
        for match in self._UPDATE_RELATIONSHIP_RE.finditer(response_text):
            relationship_text = match.group(1).strip()
            if relationship_text:
                try:
                    self.core.update_relationship(relationship_text)
                    logger.info(f"🧠 Self-edit UPDATE_RELATIONSHIP: {relationship_text[:60]}")
                except Exception as e:
                    logger.warning(f"⚠️  UPDATE_RELATIONSHIP failed: {e}")

        # Extract and execute FORGET tags
        for match in self._FORGET_RE.finditer(response_text):
            category = match.group(1).strip()
            key = match.group(2).strip()
            if category and key:
                try:
                    self.core.forget_fact(category, key)
                except Exception as e:
                    logger.warning(f"⚠️  FORGET failed: {e}")

        # Extract and execute MOMENT tags (save emotional scenes)
        for match in self._MOMENT_RE.finditer(response_text):
            moment = match.group(1).strip()
            if moment:
                try:
                    self.core.add_moment(moment)
                    logger.info(f"💕 Self-edit MOMENT saved: {moment[:60]}")
                except Exception as e:
                    logger.warning(f"⚠️  MOMENT failed: {e}")

        # Strip all tags from the response
        cleaned = self._ALL_TAGS_RE.sub('', response_text).strip()
        return cleaned

    @staticmethod
    def strip_tags(text: str) -> str:
        """Remove all self-edit tags from text (for TTS/display)."""
        return SelfEditEngine._ALL_TAGS_RE.sub('', text).strip()


# ═══════════════════════════════════════════════════════════════════════════
# Main Memory Manager — Orchestrates All Layers
# ═══════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Unified 5-layer memory manager:
      Layer 1: Core Memory (always in context)
      Layer 2: Recall Memory (conversation buffer)
      Layer 3: Archival Memory (structured facts in core_memory.json)
      Layer 4: Semantic Memory (Mem0 vector search)
      Layer 5: Self-Editing Memory (LLM updates its own memory)
    """

    def __init__(self):
        # Layer 1 + 3: Core + Archival memory
        self.core = CoreMemory()

        # Layer 2: Recall memory
        self.recall = RecallMemory()

        # Layer 5: Self-edit engine
        self.self_edit = SelfEditEngine(self.core)

        # Layer 4: Mem0 semantic memory
        self._mem0 = None
        self._mem0_ready = False
        self._lock = threading.Lock()
        
        # Layer 3: Neo4j knowledge graph
        self.neo4j = Neo4jMemory()

        # Initialize Mem0 in background
        thread = threading.Thread(target=self._init_mem0, daemon=True)
        thread.start()

    def _init_mem0(self):
        """Initialize Mem0 in background thread."""
        try:
            from mem0 import Memory
            self._mem0 = Memory.from_config(config.MEM0_CONFIG)
            self._mem0_ready = True
            logger.info("✅ Mem0 semantic memory initialized.")
        except ImportError:
            logger.warning(
                "⚠️  mem0ai not installed. Run: pip install mem0ai\n"
                "   Falling back to core + recall memory only."
            )
        except Exception as e:
            logger.warning(f"⚠️  Mem0 init failed: {e}. Using core + recall only.")

    # ─── Context building (what the LLM sees) ────────────────────────────

    def get_context_for_prompt(self, user_query: str = "") -> str:
        """
        Letta-style pre-flight retrieval logic.
        Combines core layers, classifies intent, and selectively pulls Mem0/Neo4j.
        """
        parts = []

        # Layer 1: Core memory blocks (persona + human + relationship)
        parts.append(self.core.get_prompt_section())

        # Layer 2: Recall memory (recent conversation history)
        recall_section = self.recall.get_prompt_section()
        if recall_section:
            parts.append(f"\n{recall_section}")

        if not user_query:
            return "\n".join(parts)

        # Pre-flight: Classify Intent
        resp = _direct_ollama_generate(
            prompt=user_query,
            system=INTENT_CLASSIFIER_SYSTEM,
            max_tokens=150
        )
        
        intent = {"named_entities": []}
        if resp:
            try:
                # Extract JSON if wrapped in markdown
                if "```" in resp:
                    start = resp.find("{")
                    end = resp.rfind("}") + 1
                    resp = resp[start:end]
                intent = json.loads(resp)
            except Exception as e:
                logger.debug(f"Intent parsing failed: {e}.")

        # Layer 4: Mem0 (Always search for context)
        if self._mem0_ready:
            relevant = self._search_mem0(user_query)
            if relevant:
                parts.append(f"\n=== EPISODIC RECALL (Mem0) ===\n{relevant}")

        # Layer 3: Neo4j (Relational/Entities)
        entities = intent.get("named_entities", [])
        if entities and self.neo4j._ready:
            neo_results = []
            for entity in entities:
                res = self.neo4j.query_graph(entity)
                if res:
                    neo_results.append(res)
            
            if neo_results:
                parts.append(f"\n=== GRAPH RECALL (Neo4j) ===\n" + "\n".join(neo_results))

        context = "\n".join(parts)

        # Truncate if too long
        if len(context) > config.MEMORY_MAX_CONTEXT_LENGTH:
            context = context[:config.MEMORY_MAX_CONTEXT_LENGTH] + "\n  ..."

        return context

    # ─── Memory storage (after each conversation turn) ───────────────────

    def store_conversation(self, user_text: str, assistant_text: str, full_response: str = None):
        """
        Store a conversation turn across all memory layers.
        Called after each interaction in the assistant pipeline.
        """
        if full_response is None:
            full_response = assistant_text

        # Layer 2: Add to recall buffer (synchronous, fast)
        self.recall.add_turn(user_text, assistant_text)

        # Increment total conversations counter
        with self._lock:
            self.core.data["total_conversations"] = (
                self.core.data.get("total_conversations", 0) + 1
            )
            self.core._save()

        # Layers 3+4: Background processing (async, slower)
        thread = threading.Thread(
            target=self._background_store,
            args=(user_text, full_response),
            daemon=True,
        )
        thread.start()

    def process_self_edits(self, response_text: str) -> str:
        """
        Process self-edit tags in the LLM response (Layer 5).
        Returns the cleaned response with tags stripped.
        """
        return self.self_edit.process_response(response_text)

    # ─── Background storage workers ──────────────────────────────────────

    def _background_store(self, user_text: str, assistant_text: str):
        """Background worker: extract triggers and write back to Mem0/Neo4j."""
        try:
            conversation = f"User: {user_text}\nAssistant: {assistant_text}"
            resp = _direct_ollama_generate(
                prompt=conversation,
                system=WRITE_BACK_SYSTEM,
                max_tokens=600
            )
            
            if not resp:
                return
                
            if "```" in resp:
                start = resp.find("{")
                end = resp.rfind("}") + 1
                resp = resp[start:end]

            if not resp.startswith("{"):
                start = resp.find("{")
                end = resp.rfind("}") + 1
                resp = resp[start:end]

            data = json.loads(resp)

            # Write to Mem0
            mem0_facts = data.get("mem0_episodic", [])
            if mem0_facts and self._mem0_ready and self._mem0 is not None:
                self._mem0.add(mem0_facts, user_id=config.MEM0_USER_ID)
                logger.info(f"🧠 Mem0 stored {len(mem0_facts)} episodic facts.")
                
            # Write to Neo4j
            neo4j_rels = data.get("neo4j_relations", [])
            if neo4j_rels and self.neo4j._ready:
                for rel in neo4j_rels:
                    try:
                        source = rel.get("source")
                        relation = rel.get("relation", "").upper()
                        target = rel.get("target")
                        if source and relation and target:
                            self.neo4j.add_relation(source, relation, target)
                    except Exception as e:
                        logger.debug(f"⚠️  Neo4j relation write failed: {e}")
                logger.info(f"🧠 Neo4j stored {len(neo4j_rels)} relations.")

        except json.JSONDecodeError:
            logger.debug(f"🧠 Write back JSON parse failed.")
        except Exception as e:
            logger.warning(f"⚠️  Write back process failed: {e}")

    def _search_mem0(self, query: str) -> str:
        """Layer 4: Semantic search over past conversations."""
        if not self._mem0_ready or self._mem0 is None:
            return ""

        try:
            results = self._mem0.search(
                query,
                user_id=config.MEM0_USER_ID,
                limit=config.MEM0_SEARCH_LIMIT,
            )

            if not results:
                return ""

            # Handle both list and dict formats
            mem_list = results
            if isinstance(results, dict):
                mem_list = results.get("results", [])

            if not mem_list:
                return ""

            memories = []
            for mem in mem_list:
                text = mem.get("memory", "") if isinstance(mem, dict) else str(mem)
                if text:
                    memories.append(f"  - {text}")

            return "\n".join(memories) if memories else ""

        except Exception as e:
            logger.debug(f"⚠️  Mem0 search failed: {e}")
            return ""
