"""Engram Extractor — structured fact + context extraction from raw text.

Uses the existing Engram enrichment pipeline first (echo, episodic index,
scene, salience, knowledge graph), then adds structured facts + context
anchors via the LLM.

Integration point: engram/memory/main.py _process_single_memory()
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from dhee.core.engram import (
    AssociativeLink,
    ContextAnchor,
    EntityRef,
    Fact,
    ProspectiveScene,
    SceneSnapshot,
    UniversalEngram,
)

logger = logging.getLogger(__name__)

# Extraction prompt for structured engram generation
_ENGRAM_EXTRACTION_PROMPT = """Extract structured memory from the following text.

TEXT:
{content}

{session_context_block}

Return a JSON object with these fields:
{{
  "context": {{
    "era": "life phase or null (e.g. 'school', 'college', 'bengaluru_work')",
    "place": "location or null",
    "place_type": "home|office|travel|school|city or null",
    "place_detail": "specific location detail or null",
    "time_absolute": "ISO date/datetime if determinable, or null",
    "time_markers": ["original temporal references from text"],
    "activity": "activity type or null (coding|meeting|travel|movie|exam|conversation|...)"
  }},
  "scene": {{
    "setting": "physical/virtual setting or null",
    "people_present": ["people mentioned or involved"],
    "self_state": "user's state/condition or null",
    "emotional_tone": "dominant emotion or null",
    "sensory_cues": ["sensory details or activity cues"]
  }},
  "facts": [
    {{
      "subject": "who/what",
      "predicate": "action/relation/property",
      "value": "object/value",
      "value_numeric": null,
      "value_unit": null,
      "time": "when, if known",
      "valid_from": "when this became true, if known",
      "valid_until": null,
      "qualifier": "additional context or null",
      "confidence": 1.0,
      "is_derived": false
    }}
  ],
  "entities": [
    {{
      "name": "entity name",
      "entity_type": "person|org|technology|location|project|tool",
      "state": "current|former|planned or null",
      "relationships": [{{"target": "other entity", "relation": "relation type"}}]
    }}
  ],
  "links": [
    {{
      "target_canonical_key": "subject|predicate|value",
      "link_type": "causal|temporal_sequence|co_occurring|emotional|elaborates",
      "direction": "forward|backward",
      "qualifier": "link description or null"
    }}
  ]
}}

Rules:
- Extract EVERY factual claim as a SEPARATE fact — "I ate pizza and pasta" = 2 facts
- Each distinct instance gets its OWN canonical_key: "subject|predicate|value" (lowercase, underscores)
- For counting queries: "visited Tokyo, Paris, London" = 3 separate visited facts, NOT one
- For knowledge updates ("switched from X to Y"): create 2 facts — old (valid_until=now) + new (valid_from=now)
- For preferences ("I prefer/like/use X"): use predicate "prefers" or "uses"
- Resolve temporal references to absolute dates when session date is provided
- value_numeric: extract numbers for prices, counts, distances, durations
- Return valid JSON only, no markdown"""


class EngramExtractor:
    """Extract structured Universal Engrams from raw text content."""

    def __init__(self, llm=None):
        """Initialize with an LLM for extraction.

        Args:
            llm: BaseLLM instance. If None, extraction returns minimal engrams.
        """
        self.llm = llm

    def extract(
        self,
        content: str,
        session_context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        existing_metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "default",
        strict_llm_failure: bool = False,
    ) -> UniversalEngram:
        """Extract structured engram with contextual anchoring.

        Uses LLM for structured extraction. Falls back to rule-based
        extraction if LLM is unavailable.
        """
        if not content or not content.strip():
            return UniversalEngram(raw_content=content, user_id=user_id)

        # Try LLM extraction first
        if self.llm:
            engram = self._extract_with_llm(
                content, session_context, user_profile, user_id
            )
            if engram:
                return engram
            if strict_llm_failure:
                raise RuntimeError("Engram LLM extraction produced no structured result")

        # Rule-based extraction
        return self._extract_rule_based(
            content, session_context, existing_metadata, user_id
        )

    def extract_batch(
        self,
        contents: List[str],
        session_context: Optional[Dict[str, Any]] = None,
        user_id: str = "default",
    ) -> List[UniversalEngram]:
        """Batch extraction for training data pipeline."""
        return [
            self.extract(c, session_context=session_context, user_id=user_id)
            for c in contents
        ]

    def _extract_with_llm(
        self,
        content: str,
        session_context: Optional[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]],
        user_id: str,
    ) -> Optional[UniversalEngram]:
        """LLM-based structured extraction. Fails fast — no retries here
        (the LLM client already has its own retry logic with shorter timeout)."""
        session_block = ""
        if session_context:
            session_block = f"SESSION CONTEXT:\n{json.dumps(session_context, default=str)}"

        prompt = _ENGRAM_EXTRACTION_PROMPT.format(
            content=content[:16000],
            session_context_block=session_block,
        )
        vocabulary = getattr(self, "vocabulary", None) or {}
        if vocabulary:
            # A reader joins on predicates. Given the words it reads, the model
            # uses them; left to itself it invents a new one per sentence.
            listed = "\n".join(
                f"- {name}: {meaning.get('what', meaning) if isinstance(meaning, dict) else meaning}"
                for name, meaning in vocabulary.items()
            )
            prompt += (
                "\n- For facts about the user, use ONLY these predicates, and skip a fact "
                "about the user that fits none of them:\n" + listed
            )

        try:
            # Temporarily increase max_tokens for extraction — the structured
            # JSON output (context + scene + facts + entities + links) often
            # exceeds 4096 tokens for sessions with many facts.
            original_max = getattr(self.llm, "max_tokens", None)
            if original_max is not None and original_max < 8192:
                self.llm.max_tokens = 8192
            try:
                response = self.llm.generate(prompt)
            finally:
                if original_max is not None:
                    self.llm.max_tokens = original_max

            if response:
                logger.debug("EngramExtractor raw LLM response (first 500 chars): %s", response[:500])
            parsed = self._parse_extraction_response(response)
            if not parsed:
                logger.debug("EngramExtractor: LLM returned unparseable response")
                return None

            engram = UniversalEngram(
                raw_content=content,
                context=ContextAnchor.from_dict(parsed.get("context", {})),
                scene=SceneSnapshot.from_dict(parsed.get("scene", {})),
                facts=[Fact.from_dict(f) for f in parsed.get("facts", [])],
                entities=[EntityRef.from_dict(e) for e in parsed.get("entities", [])],
                links=[
                    AssociativeLink.from_dict(l)
                    for l in parsed.get("links", [])
                ],
                user_id=user_id,
            )
            logger.info(
                "EngramExtractor: extracted %d facts, %d entities, context=%s",
                len(engram.facts), len(engram.entities),
                "yes" if engram.context.has_context() else "no",
            )
            return engram
        except Exception as e:
            logger.warning("EngramExtractor: LLM extraction failed: %s", e)
            return None

    def _extract_rule_based(
        self,
        content: str,
        session_context: Optional[Dict[str, Any]],
        existing_metadata: Optional[Dict[str, Any]],
        user_id: str,
    ) -> UniversalEngram:
        """Rule-based extraction without LLM."""
        context = self._extract_context_rules(content, session_context)
        scene = self._extract_scene_rules(content)
        facts = self._extract_facts_rules(content)
        entities = self._extract_entities_rules(content)
        prospective = self._extract_prospective_scenes(content, context, scene)

        return UniversalEngram(
            raw_content=content,
            context=context,
            scene=scene,
            facts=facts,
            entities=entities,
            prospective_scenes=prospective,
            user_id=user_id,
            metadata=existing_metadata or {},
        )

    def _extract_context_rules(
        self,
        content: str,
        session_context: Optional[Dict[str, Any]],
    ) -> ContextAnchor:
        """Rule-based context anchor extraction."""
        lower = content.lower()
        ctx = ContextAnchor()

        # Activity detection
        activity_patterns = {
            "coding": r"\b(code|coding|programming|debug|commit|deploy|refactor)\b",
            "meeting": r"\b(meeting|standup|call|sync|discussion|1:1)\b",
            "travel": r"\b(travel|trip|flight|hotel|visited|touring)\b",
            "movie": r"\b(movie|film|watched|cinema|theater)\b",
            "exam": r"\b(exam|test|quiz|assessment|grade|marks)\b",
            "reading": r"\b(read|reading|book|article|paper)\b",
            "cooking": r"\b(cook|cooking|recipe|kitchen|meal)\b",
            "exercise": r"\b(gym|workout|exercise|run|running|yoga)\b",
        }
        for activity, pattern in activity_patterns.items():
            if re.search(pattern, lower):
                ctx.activity = activity
                break

        # Place type detection
        place_patterns = {
            "office": r"\b(office|workplace|desk|coworking)\b",
            "home": r"\b(home|house|apartment|flat)\b",
            "school": r"\b(school|class|classroom|college|university)\b",
            "travel": r"\b(airport|station|hotel|flight)\b",
        }
        for place_type, pattern in place_patterns.items():
            if re.search(pattern, lower):
                ctx.place_type = place_type
                break

        # Time marker extraction
        time_patterns = [
            r"\b(yesterday|today|tomorrow)\b",
            r"\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\b(in\s+\d{4})\b",
            r"\b(class\s+\d+)\b",
            r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, lower)
            for match in matches:
                marker = match if isinstance(match, str) else " ".join(match)
                if marker.strip():
                    ctx.time_markers.append(marker.strip())

        # Session context propagation
        if session_context:
            if not ctx.era and session_context.get("era"):
                ctx.era = session_context["era"]
            if session_context.get("session_id"):
                ctx.session_id = session_context["session_id"]

        return ctx

    def _extract_scene_rules(self, content: str) -> SceneSnapshot:
        """Rule-based scene snapshot extraction."""
        scene = SceneSnapshot()
        lower = content.lower()

        # Emotional tone detection
        tone_patterns = {
            "happy": r"\b(happy|glad|excited|joyful|great|awesome|wonderful)\b",
            "sad": r"\b(sad|unhappy|disappointed|upset|terrible)\b",
            "anxious": r"\b(anxious|worried|nervous|stressed|tense)\b",
            "relaxed": r"\b(relaxed|calm|peaceful|comfortable)\b",
            "frustrated": r"\b(frustrated|annoyed|irritated|angry)\b",
        }
        for tone, pattern in tone_patterns.items():
            if re.search(pattern, lower):
                scene.emotional_tone = tone
                break

        # People extraction (simple pattern)
        people_pattern = r"\bwith\s+(\w+(?:\s+\w+)?)\b"
        people_matches = re.findall(people_pattern, lower)
        stop_words = {"the", "a", "an", "my", "some", "all", "this", "that"}
        for person in people_matches[:5]:
            if person.strip() not in stop_words:
                scene.people_present.append(person.strip())

        return scene

    def _extract_facts_rules(self, content: str) -> List[Fact]:
        """Rule-based fact extraction — deterministic patterns."""
        facts = []

        # Pattern: "X prefers Y" / "X likes Y" / "X uses Y"
        preference_patterns = [
            (r"(?:I|user)\s+(?:prefer|like|love|enjoy|use)\s+(.+?)(?:\.|$|,)", "prefers"),
            (r"(?:I|user)\s+(?:switched to|moved to|changed to)\s+(.+?)(?:\.|$|,)", "switched_to"),
            (r"(?:I|user)\s+(?:work at|work for|employed at)\s+(.+?)(?:\.|$|,)", "works_at"),
            (r"(?:I|user|owner)\s+(?:live|lives|reside|resides)\s+in\s+(.+?)(?:\.|$|,)", "lives_in"),
            (r"(?:my|user's|owner's)\s+(?:home|city|location)\s+is\s+(.+?)(?:\.|$|,)", "lives_in"),
            (r"(?:my|user's|owner's)\s+(?:plan|goal)\s+is\s+to\s+(.+?)(?:\.|$|,)", "plans_to"),
            (r"(?:I|user|owner)\s+(?:plan|plans|intend|intends)\s+to\s+(.+?)(?:\.|$|,)", "plans_to"),
            (r"(?:I|user)\s+(?:visited|went to|traveled to)\s+(.+?)(?:\.|$|,)", "visited"),
            (r"(?:I|user)\s+(?:bought|purchased)\s+(.+?)(?:\s+for\s+)?([\d,.]+)?\s*(\w+)?", "bought"),
        ]
        for pattern, predicate in preference_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                value = self._clean_rule_value(match.group(1))
                if len(value) > 100:
                    continue
                fact = Fact(
                    subject="user",
                    predicate=predicate,
                    value=value,
                )
                fact.canonical_key = fact.make_canonical_key()

                # Extract numeric values for "bought" pattern
                if predicate == "bought" and match.lastindex and match.lastindex >= 2:
                    try:
                        numeric = match.group(2)
                        if numeric:
                            fact.value_numeric = float(numeric.replace(",", ""))
                            fact.value_unit = match.group(3) if match.lastindex >= 3 else None
                    except (ValueError, IndexError):
                        pass

                facts.append(fact)

        return facts

    @staticmethod
    def _clean_rule_value(value: str) -> str:
        text = str(value or "").strip().rstrip(".")
        text = re.split(
            r"\s+(?:and|but|while)\s+(?=(?:I|user|owner|my|user's|owner's)\b)",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return text.strip().rstrip(".")

    def _extract_entities_rules(self, content: str) -> List[EntityRef]:
        """Rule-based entity extraction."""
        entities = []

        # Capitalized words as potential entities (simple heuristic)
        words = content.split()
        seen = set()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:'\"()[]")
            if (
                clean
                and clean[0].isupper()
                and len(clean) > 1
                and clean.lower() not in {"i", "the", "a", "an", "my", "we", "he", "she", "it", "they"}
                and clean not in seen
            ):
                seen.add(clean)
                entities.append(EntityRef(name=clean, entity_type="unknown"))

        return entities[:10]  # Cap at 10 entities

    def _extract_prospective_scenes(
        self,
        content: str,
        context: ContextAnchor,
        scene: SceneSnapshot,
    ) -> List[ProspectiveScene]:
        """Detect future intent and create ProspectiveScene predictions.

        When someone says "we'll play tennis next Saturday with Ankit",
        the memory engine creates a predicted scene — NOT a todo.
        It links to past similar scenes and predicts what you'll need.
        """
        lower = content.lower()
        prospective = []

        # Future intent patterns
        future_patterns = [
            # "will/going to/plan to [verb] [with person] [time]"
            r"(?:will|going to|plan(?:ning)? to|gonna|shall|let's|we(?:'ll| will))\s+(.+?)(?:\.|$)",
            # "next [day/week/month]"
            r"(?:next|coming|upcoming|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|weekend)",
            # "[day] we have/there's a [event]"
            r"(?:on|this|next)\s+\w+\s+(?:we have|there(?:'s| is)|I have)\s+(.+?)(?:\.|$)",
            # "meeting/appointment/call with [person] on/at [time]"
            r"(?:meeting|appointment|call|session|game|match|dinner|lunch)\s+(?:with\s+)?(\w+)\s+(?:on|at|next|this)",
        ]

        has_future_intent = False
        future_text = ""
        for pattern in future_patterns:
            match = re.search(pattern, lower)
            if match:
                has_future_intent = True
                future_text = match.group(0)
                break

        if not has_future_intent:
            return []

        # Extract participants (people mentioned with "with")
        participants = []
        with_match = re.search(r"\bwith\s+(\w+(?:\s+(?:and|&)\s+\w+)*)", lower)
        if with_match:
            names = re.split(r"\s+(?:and|&)\s+", with_match.group(1))
            participants = [n.strip().title() for n in names if n.strip()]

        # Also use scene's people_present
        if scene.people_present:
            for p in scene.people_present:
                if p.title() not in participants:
                    participants.append(p.title())

        # Detect event type
        event_type = None
        event_patterns = {
            "sport": r"\b(tennis|cricket|football|soccer|badminton|gym|yoga|run|swim|play)\b",
            "meeting": r"\b(meeting|standup|sync|call|discussion|1:1|review)\b",
            "social": r"\b(dinner|lunch|party|hangout|movie|drinks|coffee|birthday)\b",
            "travel": r"\b(trip|travel|flight|visit|tour)\b",
            "deadline": r"\b(deadline|due|submit|deliver|launch|release)\b",
        }
        for etype, pattern in event_patterns.items():
            if re.search(pattern, lower):
                event_type = etype
                break

        # Extract predicted time (simplified — full resolution would use DheeModel)
        predicted_time = None
        time_match = re.search(
            r"(?:next|coming|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            lower,
        )
        if time_match:
            from datetime import datetime, timedelta, timezone
            day_name = time_match.group(1).lower()
            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6,
            }
            target_day = day_map.get(day_name, 0)
            now = datetime.now(timezone.utc)
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            target_date = now + timedelta(days=days_ahead)
            predicted_time = target_date.replace(hour=10, minute=0, second=0).isoformat()

        if not time_match:
            # "next month's 1st saturday" pattern
            month_match = re.search(
                r"next\s+month(?:'s)?\s+(?:1st|first|2nd|second|3rd|third|last)\s+"
                r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                lower,
            )
            if month_match:
                from datetime import datetime, timedelta, timezone
                from calendar import monthrange
                day_name = month_match.group(1).lower()
                day_map = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6,
                }
                target_day = day_map.get(day_name, 5)
                now = datetime.now(timezone.utc)
                next_month = now.month + 1 if now.month < 12 else 1
                next_year = now.year if now.month < 12 else now.year + 1
                # Find ordinal occurrence
                ordinal = "1st"
                if "2nd" in lower or "second" in lower:
                    ordinal = "2nd"
                elif "3rd" in lower or "third" in lower:
                    ordinal = "3rd"
                elif "last" in lower:
                    ordinal = "last"

                # Find the Nth occurrence of the day in next month
                first_of_month = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
                days_in_month = monthrange(next_year, next_month)[1]
                occurrences = []
                for d in range(1, days_in_month + 1):
                    dt = datetime(next_year, next_month, d, tzinfo=timezone.utc)
                    if dt.weekday() == target_day:
                        occurrences.append(dt)
                if occurrences:
                    if ordinal == "1st":
                        predicted_time = occurrences[0].replace(hour=10).isoformat()
                    elif ordinal == "2nd" and len(occurrences) >= 2:
                        predicted_time = occurrences[1].replace(hour=10).isoformat()
                    elif ordinal == "3rd" and len(occurrences) >= 3:
                        predicted_time = occurrences[2].replace(hour=10).isoformat()
                    elif ordinal == "last":
                        predicted_time = occurrences[-1].replace(hour=10).isoformat()

        prospective.append(ProspectiveScene(
            predicted_time=predicted_time,
            trigger_window_hours=24,
            event_type=event_type,
            participants=participants,
            predicted_setting=scene.setting,
            predicted_needs=[],  # Will be enriched with past scene data at storage time
            status="predicted",
            prediction_basis=f"Detected future intent: '{future_text}'",
        ))

        return prospective

    def _parse_extraction_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM extraction response, handling markdown code blocks and truncation."""
        if not response:
            return None

        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Handle truncated JSON (finish_reason=length) — salvage partial data.
        # The model often produces valid context/scene/facts before running out of tokens.
        if brace_start >= 0:
            truncated = text[brace_start:]
            repaired = self._repair_truncated_json(truncated)
            if repaired:
                logger.info("EngramExtractor: salvaged partial JSON from truncated response")
                return repaired

        logger.warning("Failed to parse LLM extraction response")
        return None

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair truncated JSON by closing open structures.

        When max_tokens is hit, the JSON is cut mid-stream. This method
        walks backward from the truncation point to find the last complete
        JSON value boundary, then closes all open arrays/objects.
        """
        # Walk backward to find a safe cut point — the end of a complete
        # JSON value (after a }, ], number, "string", true, false, null)
        # followed by optional whitespace and a comma or container-close.
        # We use a regex to find the last "}, " or "}," or "}]" boundary.
        cut_re = re.compile(r'(}|])\s*[,\]}\n]')
        matches = list(cut_re.finditer(text))
        if not matches:
            return None

        # Try from the latest cut point backward
        for match in reversed(matches):
            cut_pos = match.end()
            fragment = text[:cut_pos].rstrip().rstrip(",")

            # Count open braces/brackets
            open_braces = fragment.count("{") - fragment.count("}")
            open_brackets = fragment.count("[") - fragment.count("]")

            # Close open structures
            suffix = "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            candidate = fragment + suffix

            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

        return None
