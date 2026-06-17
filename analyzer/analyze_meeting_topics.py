"""
Deterministic municipal meeting analyzer.

Analyzes transcripts of municipal council meetings to extract topics and
generate structured summaries. Uses a deterministic NLP pipeline rather than
relying solely on LLM interpretation.

Pipeline:
    1. Load transcript → parse timestamped speaker text
    2. Merge utterances → combine consecutive statements by same speaker
    3. Build segments → create overlapping time windows
    4. Lemmatize → reduce words to base forms (Czech language)
    5. TF-IDF → identify important words per segment
    6. Cluster → group segments into topics using HDBSCAN
    7. Summarize → extract topic metadata and representative sentences
    8. Format → prepare structured input for LLM article generation

The deterministic approach ensures the LLM receives focused, structured data
rather than making subjective decisions about what's important.
"""

import argparse
import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import ufal.morphodita as morphodita



# =========================
# CONFIG
# =========================

# Model and paths
SCRIPT_DIR = Path(__file__).resolve().parent
MORPHODITA_MODEL = str(SCRIPT_DIR / "czech-morfflex2.1-pdtc2.0-250909.tagger")

# Segmentation parameters
SEGMENT_LEN = 300      # Segment length in seconds (5 minutes)
SEGMENT_OVERLAP = 120  # Overlap between segments in seconds (2 minutes)
MERGE_GAP = 5          # Maximum gap in seconds to merge consecutive utterances from same speaker

# Clustering parameters
MIN_TOPIC_SIZE = 2     # Minimum number of segments required to form a topic cluster

# Keyword filtering - common verbs and generic words to exclude from topics
UNWANTED_KEYWORDS = {
    "být", "mít", "říci", "říkat", "chtít", "moci",
    "dělat", "vědět", "myslit", "jít", "prosit", 
    "udělat", "řešit", "mluvit", "věc"
}

# Transcription error corrections - common speech-to-text mistakes specific to this locale
COMMON_MISTAKES = {
    "písavný": "písemné",
    "Litovla": "Litovle",
    "Litového": "Litovel",
    "Litovl": "Litovel",
    "Litovélo": "Litovel",
    "Stavěnový": "Stavební",
    "Stavěvní": "Stavební",
    "navědomý": "na vědomí",
    "zápisě": "zápise",
    "krátkrobým": "krátkodobým",
    "Ritovel": "Litovel",
    "rozpoštením": "rozpočtovým",
    "Litovilsko": "Litovelsko",
    "na Sovburgách": "v Nasobůrkách",
    "psířiště": "psí hřiště",
    " krum": " korun",
    "zasadeny": "zasazeny",
    "litovaské": "litovelské",
    "po zemku": "pozemku",
    "dobudové": "důvodové",
    "Litovelezero": "Litovel s.r.o.",
    "Žejrenko": "", #TODO
    "řezové": "Březové",
    "Alomouckem": "Olomouckém",
    "Alomoucko": "Olomouckou",
    "Alomouckej": "Olomoucké",
    " toveláci": " litoveláci"
}

# Domain hints - map lemmas to topic categories for better topic labeling
DOMAIN_HINTS = {
    "stavba": "průběh stavby",
    "silnice": "místní komunikace",
    "výkop": "stavební práce",
    "vodovod": "vodovodní infrastruktura",
    "kanalizace": "kanalizace",
    "dotace": "dotace a financování",
    "obyvatel": "dopad na obyvatele",
    "komunikace": "komunikace města s občany",
    "kontrola": "kontrola a dohled",
    "usnesení": "postup orgánů města",
    "pozemek": "majetek města",
    "škola": "školství",
    "mikroregion": "meziobecní spolupráce",
}

# Hint keywords - used to find representative sentences that match topic hints
HINT_KEYWORDS = {
    "školství": ["škola", "školy", "školní", "žák", "učitel"],
    "vodovodní infrastruktura": ["vodovod", "přípojka", "voda"],
    "kanalizace": ["kanalizace", "kanál"],
    "místní komunikace": ["silnice", "chodník", "cesta"],
    "průběh stavby": ["stavba", "výkop", "projekt"],
}

# =========================
# HELPERS
# =========================

def parse_time(t):
    """
    Convert timestamp string to seconds.

    Args:
        t: Time string in format "HH:MM:SS"

    Returns:
        int: Total seconds

    Example:
        parse_time("1:30:45") -> 5445
    """
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def _segment_to_json(row):
    """
    Convert a segment DataFrame row to a JSON-serializable dictionary.

    Args:
        row: pandas Series with segment data

    Returns:
        dict: Segment data with id, timing, speakers, and text information
    """
    return {
        "id": int(row.name),
        "start": float(row.t_start),
        "end": float(row.t_end),
        "duration": float(row.t_end - row.t_start),
        "speakers": list(row.speakers),
        "speaker_count": len(row.speakers),
        "speaker_texts": {
            speaker: text
            for speaker, text in row.speaker_texts.items()
        },
        "word_count": int(
            sum(len(txt.split()) for txt in row.speaker_texts.values())
        )
    }

def _jaccard_similarity(a, b):
    """
    Calculate Jaccard similarity between two sentences.

    Jaccard similarity is the size of intersection divided by the size of union.
    Measures how similar two sentences are based on shared words.

    Args:
        a: First sentence string
        b: Second sentence string

    Returns:
        float: Similarity score between 0.0 (no overlap) and 1.0 (identical)

    Example:
        _jaccard_similarity("hello world", "hello there") -> 0.333...
        # intersection: {hello} (1 word)
        # union: {hello, world, there} (3 words)
        # score: 1/3
    """
    A = set(a.lower().split())
    B = set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def _find_hint_sentence(sentences, hint):
    """
    Find the first sentence that contains keywords related to a topic hint.

    Args:
        sentences: List of sentence strings to search
        hint: Topic hint string (must be a key in HINT_KEYWORDS)

    Returns:
        str or None: First matching sentence, or None if no match found
    """
    if not hint or hint not in HINT_KEYWORDS:
        return None

    keywords = HINT_KEYWORDS[hint]

    for s in sentences:
        s_l = s.lower()
        if any(k in s_l for k in keywords):
            return s

    return None

# =========================
# STEP 1: LOAD TRANSCRIPT
# =========================

def load_transcript(path) -> pd.DataFrame:
    """
    Load and parse a transcript file into a DataFrame.

    Parses transcript with format:
        [HH:MM:SS] SPEAKER_NAME:
        Text spoken by speaker...

    Also applies common transcription error corrections from COMMON_MISTAKES.

    Args:
        path: Path to transcript file

    Returns:
        pd.DataFrame: DataFrame with columns:
            - t: timestamp in seconds (int)
            - speaker: speaker name (str)
            - text: utterance text (str)
    """
    rows = []

    time_rx = re.compile(r"\[(\d+:\d+:\d+)\]\s+(\w+):")

    current_t = None
    current_speaker = None

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Apply common transcription error corrections
        for wrong in sorted(COMMON_MISTAKES, key=len, reverse=True):
            line = line.replace(wrong, COMMON_MISTAKES[wrong])

        # Check if line is a timestamp header
        m = time_rx.match(line)
        if m:
            current_t = parse_time(m.group(1))
            current_speaker = m.group(2)
            continue

        # Add text line to current speaker's utterance
        if current_t is not None and line:
            rows.append({
                "t": current_t,
                "speaker": current_speaker,
                "text": line
            })

    df = pd.DataFrame(rows)
    return df


# =========================
# STEP 2: merge items said by the same speaker inside the MERGE_GAP
# =========================

def merge_utterances(df) -> pd.DataFrame:
    """
    Merge consecutive utterances from the same speaker.

    Combines utterances from the same speaker if they occur within MERGE_GAP seconds.
    This reduces fragmentation in the transcript.

    Args:
        df: DataFrame from load_transcript() with columns: t, speaker, text

    Returns:
        pd.DataFrame: Merged utterances with columns:
            - speaker: speaker name (str)
            - t_start: start timestamp in seconds (int)
            - t_end: end timestamp in seconds (int)
            - text: combined utterance text (str)
            - word_count: number of words (int)
    """
    merged = []

    current = None

    for row in df.itertuples():
        # Merge if same speaker and within MERGE_GAP seconds
        if (
            current
            and row.speaker == current["speaker"]
            and row.t - current["t_end"] <= MERGE_GAP
        ):
            current["text"] += " " + row.text
            current["t_end"] = row.t
        else:
            # Start new utterance
            if current:
                merged.append(current)
            current = {
                "speaker": row.speaker,
                "t_start": row.t,
                "t_end": row.t,
                "text": row.text,
                "word_count": len(row.text.split())
            }

    if current:
        merged.append(current)

    return pd.DataFrame(merged)


# =========================
# STEP 3: divide to segments SEGMENT_LEN long with SEGMENT_OVERLAP
# =========================

def build_segments(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """
    Divide transcript into overlapping time-based segments.

    Creates segments of SEGMENT_LEN seconds with SEGMENT_OVERLAP seconds overlap.
    This allows the clustering algorithm to better identify topics that span time windows.

    Args:
        df: DataFrame from merge_utterances() with t_start, t_end, speaker, text
        outdir: Output directory where segments.json will be saved

    Returns:
        pd.DataFrame: Segments with columns:
            - t_start: segment start time in seconds (int)
            - t_end: segment end time in seconds (int)
            - index: segment number (int)
            - speakers: list of speaker names in segment (list[str])
            - speaker_texts: dict mapping speaker to combined text (dict)
            - word_count: total words in segment (int)

    Side effects:
        Writes segments.json to outdir
    """
    t_min = df["t_start"].min()
    t_max = df["t_end"].max()

    segments = []
    t = t_min

    index = 1
    while t < t_max:
        t_end = t + SEGMENT_LEN

        # Select utterances that overlap with this time window
        mask = (df["t_end"] >= t) & (df["t_start"] <= t_end)
        chunk = df[mask]

        if not chunk.empty:
            segments.append({
                "t_start": int(t),
                "t_end": int(t_end),
                "index": index,
                "speakers": list(set(chunk["speaker"])),
                "speaker_texts": (
                    chunk.groupby("speaker")["text"]
                    .apply(lambda x: " ".join(x))
                    .to_dict()
                ),
                "word_count": int(chunk["text"].str.split().str.len().sum())
            })
            index = index + 1

        # Advance with overlap
        t += SEGMENT_LEN - SEGMENT_OVERLAP

    # Save segments to file
    Path(outdir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return pd.DataFrame(segments)


# =========================
# STEP 4: MorphoDiTa LEMMATIZATION - NLP technique to find word base forms
# =========================

class Lemmatizer:
    """
    Czech language lemmatizer using MorphoDiTa.

    Lemmatization converts words to their base forms (e.g., "školy" -> "škola").
    This reduces noise and helps identify topics more accurately.

    Only extracts nouns (N), verbs (V), and adjectives (A) as they carry
    the most semantic meaning for topic detection.
    """

    def __init__(self, model_path):
        """
        Initialize the lemmatizer with a MorphoDiTa model.

        Args:
            model_path: Path to MorphoDiTa .tagger model file

        Raises:
            RuntimeError: If model cannot be loaded
        """
        self.tagger = morphodita.Tagger.load(model_path)
        if not self.tagger:
            raise RuntimeError("Cannot load MorphoDiTa model")

        self.tokenizer = self.tagger.newTokenizer()
        self.forms = morphodita.Forms()
        self.lemmas = morphodita.TaggedLemmas()
        self.tokens = morphodita.TokenRanges()

    def lemmatize(self, text):
        """
        Convert text to list of base-form lemmas.

        Filters to only nouns, verbs, and adjectives (based on POS tags).

        Args:
            text: Czech text to lemmatize

        Returns:
            list[str]: List of lowercase lemmas (base forms of words)

        Example:
            lemmatize("Ve školách jsou žáci") -> ["škola", "být", "žák"]
        """
        self.tokenizer.setText(text)
        lemmas_out = []

        while self.tokenizer.nextSentence(self.forms, self.tokens):
            self.tagger.tag(self.forms, self.lemmas)

            for lemma in self.lemmas:
                base = lemma.lemma.split("_")[0]
                tag = lemma.tag

                # Filter by POS (part-of-speech): N=noun, V=verb, A=adjective
                if tag[0] in {"N", "V", "A"} and base.isalpha():
                    lemmas_out.append(base.lower())

        return lemmas_out


# =========================
# STEP 5: TF-IDF - use statistics to determine which words are the most important ones
# =========================

def build_tfidf(segments: pd.DataFrame, lemmatizer: Lemmatizer):
    """
    Build TF-IDF vectors from segments for clustering.

    TF-IDF (Term Frequency-Inverse Document Frequency) identifies words that
    are important to a segment but not common across all segments.

    Process:
    1. Lemmatize all segment text
    2. Build TF-IDF matrix where each row is a segment vector
    3. Words appearing in >95% of segments are filtered (too common)

    Args:
        segments: DataFrame from build_segments()
        lemmatizer: Lemmatizer instance for text processing

    Returns:
        tuple: (X, vectorizer, lemma_maps) where:
            - X: scipy sparse matrix of TF-IDF vectors (n_segments × n_features)
            - vectorizer: fitted TfidfVectorizer for feature inspection
            - lemma_maps: list of Counter objects with lemma frequencies per segment
    """
    docs = []
    lemma_maps = []

    # Lemmatize each segment
    for seg in tqdm(segments.itertuples(), total=len(segments)):
        full_text = " ".join(seg.speaker_texts.values())

        lemmas = lemmatizer.lemmatize(full_text)

        lemma_maps.append(Counter(lemmas))
        docs.append(" ".join(lemmas))

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(
        min_df=1,        # minimum document frequency
        max_df=0.95,     # filter words appearing in >95% of segments
        ngram_range=(1, 1)  # use single words only
    )

    X = vectorizer.fit_transform(docs)
    return X, vectorizer, lemma_maps

# =========================
# STEP 6: CLUSTERING
# =========================

def cluster_segments(X):
    """
    Cluster segments into topics using HDBSCAN.

    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    automatically finds topics without requiring a predetermined number of clusters.
    Segments that don't fit any cluster are marked as noise (label = -1).

    Args:
        X: TF-IDF matrix from build_tfidf() (sparse matrix)

    Returns:
        np.array: Cluster labels for each segment
            - labels >= 0: topic/cluster ID
            - label = -1: noise (doesn't belong to any topic)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X.toarray())
    return labels


# =========================
# STEP 7: AGGREGATE TOPICS
# =========================

def _extract_time_range(segs):
    """Calculate time range for a topic."""
    t_start = min(seg.t_start for seg in segs)
    t_end = max(seg.t_end for seg in segs)
    time_minutes = round((t_end - t_start) / 60, 1)
    return t_start, t_end, time_minutes


def _compute_top_lemmas(idxs, lemma_maps):
    """Extract and filter top lemmas from segments."""
    lemma_counter = Counter()
    for i in idxs:
        lemma_counter.update(lemma_maps[i])

    top_lemmas = [
        lemma for lemma, _ in lemma_counter.most_common(20)
        if lemma not in UNWANTED_KEYWORDS
    ][:15]

    return top_lemmas


def _analyze_speakers(segs):
    """Calculate speaker statistics and determine dominant speaker."""
    speaker_words = Counter()
    for seg in segs:
        for speaker, text in seg.speaker_texts.items():
            speaker_words[speaker] += len(text.split())

    total_words = sum(speaker_words.values())
    dominant_ratio = max(speaker_words.values()) / total_words if total_words > 0 else 0
    speaker_count = len(speaker_words)
    dominant_speaker = speaker_words.most_common(1)[0][0] if speaker_words else None

    return {
        "speaker_words": speaker_words,
        "total_words": total_words,
        "dominant_ratio": dominant_ratio,
        "speaker_count": speaker_count,
        "dominant_speaker": dominant_speaker
    }


def _determine_topic_type(speaker_stats):
    """Determine if topic is monologue, discussion, or procedural."""
    dominant_ratio = speaker_stats["dominant_ratio"]
    speaker_count = speaker_stats["speaker_count"]

    if dominant_ratio > 0.75 and speaker_count <= 3:
        return "monologue"
    elif speaker_count >= 3:
        return "discussion"
    else:
        return "procedural"


def _generate_topic_hint(top_lemmas):
    """Generate topic hint from top lemmas using domain hints."""
    topic_hint = sorted({
        DOMAIN_HINTS[lemma] for lemma in top_lemmas if lemma in DOMAIN_HINTS
    })
    return ", ".join(topic_hint)


def _score_sentence(sentence, top_lemmas):
    """
    Score a sentence based on topic relevance and length quality.

    The score combines two factors:
    1. Relevance: How many topic keywords (lemmas) appear in the sentence
    2. Quality: Length penalty to avoid very short sentences (< 20 words get penalized)

    Formula: relevance_score * length_quality_factor
    - relevance_score: count of top_lemmas found in sentence
    - length_quality_factor: min(word_count / 20, 1.0)
      - sentences with 20+ words get factor of 1.0 (no penalty)
      - shorter sentences get proportional penalty (e.g., 10 words → 0.5 factor)
    """
    # Count how many topic keywords appear in this sentence
    lemma_match_count = sum(1 for lemma in top_lemmas if lemma in sentence.lower())

    # Calculate length quality factor (penalize sentences shorter than 20 words)
    word_count = len(sentence.split())
    length_quality_factor = min(word_count / 20.0, 1.0)

    # Final score: relevance weighted by length quality
    score = lemma_match_count * length_quality_factor

    return score


def _seconds_to_minutes(seconds):
    """Convert seconds to minutes rounded to one decimal place."""
    return round(seconds / 60.0, 1)


def _format_minute_range(start_seconds, end_seconds):
    """Format topic time range for prompt-friendly output."""
    return f"{_seconds_to_minutes(start_seconds):.1f}–{_seconds_to_minutes(end_seconds):.1f} min"


def _classify_evidence_type(sentence):
    """Classify sentence role with cheap deterministic heuristics."""
    lowered = sentence.lower()

    if any(token in lowered for token in [
        "schvál", "souhlas", "odsouhlas", "usnesen", "pověř", "ulož",
        "rozhod", "výběr", "vybrán", "odhlas"
    ]):
        return "decision"

    if any(token in lowered for token in [
        "obava", "problém", "stíž", "krit", "nesouhlas", "rizik",
        "vadí", "upozorn", "otázk", "pochyb"
    ]):
        return "concern"

    if any(token in lowered for token in [
        "protože", "důvod", "znamen", "vysvětl", "kvůli", "aby",
        "jde o", "zahrnuje", "počítá se"
    ]):
        return "explanation"

    if any(token in lowered for token in [
        "program", "zápis", "ověřovatel", "procedur", "hlasování",
        "bod", "jednací", "navržený program"
    ]):
        return "procedural"

    return "discussion"


def _sentence_information_score(sentence):
    """Detect concrete factual anchors such as numbers, places, or dates."""
    score = 0.0

    if re.search(r"\b\d+[.,]?\d*\b", sentence):
        score += 1.2

    if re.search(r"\b(Kč|korun|milion|mil\.|tisíc|procent|%)\b", sentence, re.IGNORECASE):
        score += 1.0

    if re.search(
        r"\b\d{1,2}\.\s*\d{1,2}\.\s*\d{2,4}\b|\b(leden|únor|březen|duben|květen|červen|červenec|srpen|září|říjen|listopad|prosinec)\b",
        sentence,
        re.IGNORECASE
    ):
        score += 0.8

    capitalized_tokens = re.findall(r"\b[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+\b", sentence)
    if len(capitalized_tokens) >= 2:
        score += 0.8

    if re.search(r"\b(v|ve|na|do|u)\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ][a-záčďéěíňóřšťúůýž]+\b", sentence):
        score += 0.6

    return score


def _collect_sentence_candidates(segs, top_lemmas, topic_hint):
    """Build scored sentence candidates with metadata from topic segments."""
    candidates = []

    for seg in sorted(segs, key=lambda seg: seg.t_start):
        ordered_texts = [seg.speaker_texts[speaker] for speaker in sorted(seg.speaker_texts)]
        for text in ordered_texts:
            for sentence in re.split(r'(?<=[.!?])\s+', text):
                sentence = sentence.strip()
                if len(sentence.split()) <= 8:
                    continue

                base_score = _score_sentence(sentence, top_lemmas)
                if base_score <= 0:
                    continue

                info_score = _sentence_information_score(sentence)
                evidence_type = _classify_evidence_type(sentence)
                score = base_score + info_score

                if topic_hint and sentence == _find_hint_sentence([sentence], topic_hint):
                    score += 0.6

                if evidence_type == "decision":
                    score += 0.5
                elif evidence_type in {"concern", "explanation"}:
                    score += 0.25

                candidates.append({
                    "text": sentence,
                    "score": round(score, 3),
                    "evidence_type": evidence_type,
                    "segment_id": int(seg.name),
                    "start_minute": _seconds_to_minutes(seg.t_start),
                    "end_minute": _seconds_to_minutes(seg.t_end),
                    "time_range": _format_minute_range(seg.t_start, seg.t_end),
                })

    unique_candidates = []
    seen_texts = set()
    for candidate in candidates:
        if candidate["text"] in seen_texts:
            continue
        seen_texts.add(candidate["text"])
        unique_candidates.append(candidate)

    return unique_candidates


def _evidence_budget(time_minutes):
    """Scale evidence count modestly with topic duration."""
    if time_minutes >= 20:
        return 5
    if time_minutes >= 10:
        return 4
    return 3


def _select_representative_evidence(candidates, topic_hint, evidence_budget):
    """
    Select representative evidence with scoring and diversity selection.

    Priorities:
    1. Keep a strong hint-matching sentence when available
    2. Preserve concrete sentences with numbers, dates, places, or decisions
    3. Spread evidence across different segments when possible
    4. Avoid near-duplicate phrasing

    Args:
        candidates: Sentence candidate metadata
        topic_hint: Domain hint string (or empty)
        evidence_budget: Maximum evidence items to keep

    Returns:
        list[dict]: Representative evidence objects
    """
    if not candidates:
        return []

    selected = []
    selected_texts = []
    seen_segments = set()

    sentences = [candidate["text"] for candidate in candidates]
    hint_sentence = _find_hint_sentence(sentences, topic_hint)
    if hint_sentence:
        for candidate in candidates:
            if candidate["text"] == hint_sentence:
                selected.append(candidate)
                selected_texts.append(candidate["text"])
                seen_segments.add(candidate["segment_id"])
                break

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["score"],
            candidate["start_minute"],
            candidate["segment_id"],
            candidate["text"]
        )
    )

    for prefer_new_segment in (True, False):
        for candidate in ranked:
            if len(selected) >= evidence_budget:
                break
            if candidate["text"] in selected_texts:
                continue
            if prefer_new_segment and candidate["segment_id"] in seen_segments:
                continue
            if any(_jaccard_similarity(candidate["text"], prev) >= 0.5 for prev in selected_texts):
                continue

            selected.append(candidate)
            selected_texts.append(candidate["text"])
            seen_segments.add(candidate["segment_id"])

    return [
        {
            "text": candidate["text"],
            "time_range": candidate["time_range"],
            "start_minute": candidate["start_minute"],
            "end_minute": candidate["end_minute"],
            "evidence_type": candidate["evidence_type"],
        }
        for candidate in selected
    ]


def _derive_discussion_intensity(time_minutes, speaker_stats, segments_count):
    """Derive a lightweight discussion-intensity signal."""
    if (
        time_minutes >= 18
        or (speaker_stats["speaker_count"] >= 4 and speaker_stats["dominant_ratio"] < 0.6)
        or segments_count >= 5
    ):
        return "high"

    if (
        time_minutes >= 8
        or speaker_stats["speaker_count"] >= 3
        or speaker_stats["dominant_ratio"] < 0.75
        or segments_count >= 3
    ):
        return "medium"

    return "low"


def _build_topic_summary_hint(topic_hint, top_lemmas, evidence):
    """Build a short deterministic hint for the article prompt."""
    hint_parts = []

    if topic_hint:
        hint_parts.append(topic_hint)

    if top_lemmas:
        hint_parts.append(f"klíčová slova: {', '.join(top_lemmas[:5])}")

    evidence_types = sorted({item["evidence_type"] for item in evidence})
    if evidence_types:
        hint_parts.append(f"typy zmínek: {', '.join(evidence_types)}")

    return "; ".join(hint_parts)


def summarize_topics(segments: pd.DataFrame, labels, lemma_maps):
    """
    Aggregate segments into topic summaries with metadata and representative text.

    Returns a list of topic dictionaries sorted by time spent on each topic.
    """
    # Group segments by cluster labels
    topics = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            topics[label].append(i)

    summaries = []

    # Process each topic
    for label, idxs in topics.items():
        segs = [segments.iloc[i] for i in idxs]

        # Extract topic features using helper functions
        t_start, t_end, time_minutes = _extract_time_range(segs)
        top_lemmas = _compute_top_lemmas(idxs, lemma_maps)
        speaker_stats = _analyze_speakers(segs)
        topic_type = _determine_topic_type(speaker_stats)
        topic_hint = _generate_topic_hint(top_lemmas)
        evidence_candidates = _collect_sentence_candidates(segs, top_lemmas, topic_hint)
        evidence_budget = _evidence_budget(time_minutes)
        representative_evidence = _select_representative_evidence(
            evidence_candidates,
            topic_hint,
            evidence_budget
        )
        discussion_intensity = _derive_discussion_intensity(
            time_minutes,
            speaker_stats,
            len(idxs)
        )
        topic_summary_hint = _build_topic_summary_hint(
            topic_hint,
            top_lemmas,
            representative_evidence
        )

        # Build final summary object
        summaries.append({
            "topic_id": int(label),
            "segments_ids": idxs,
            "segments_count": len(idxs),
            "segments": [_segment_to_json(segments.iloc[i]) for i in idxs],
            "speakers": sorted(speaker_stats["speaker_words"].keys()),
            "speaker_count": speaker_stats["speaker_count"],
            "dominant_speaker_ratio": round(speaker_stats["dominant_ratio"], 2),
            "start_minute": _seconds_to_minutes(t_start),
            "end_minute": _seconds_to_minutes(t_end),
            "time_range": _format_minute_range(t_start, t_end),
            "time_minutes": time_minutes,
            "topic_type": topic_type,
            "topic_hint": topic_hint,
            "top_lemmas": top_lemmas,
            "discussion_intensity": discussion_intensity,
            "topic_summary_hint": topic_summary_hint,
            "representative_evidence": representative_evidence,
            "representative_text": [item["text"] for item in representative_evidence]
        })

    # Sort by time spent (most to least)
    return sorted(summaries, key=lambda x: -x["time_minutes"])    


def build_llm_query_payload(
    topics,
    min_minutes=2.0,
    max_topics=12,
    max_priority_topics=3):
    """
    Prepare structured input for LLM from topic summaries.

    Filters and formats topics into a deterministic article brief:
    1. Removes short/insignificant topics (< min_minutes)
    2. Limits total topic count
    3. Keeps topic list in reading order (chronological)
    4. Adds meeting-level context and priority topics

    This creates a dense article brief that gives the LLM enough concrete
    material to write readable coverage without inventing detail.

    Args:
        topics: List of topic dicts from summarize_topics()
        min_minutes: Minimum topic duration to include (default: 2.0)
        max_topics: Maximum number of topics to include (default: 12)
        max_priority_topics: Maximum priority topics for lead guidance

    Returns:
        dict: Deterministic article brief with meeting overview and topics
    """

    filtered = [
        t for t in topics
        if t.get("time_minutes", 0) >= min_minutes
    ]

    ranked_topics = sorted(
        filtered,
        key=lambda topic: (
            -topic["time_minutes"],
            -len(topic.get("representative_evidence", [])),
            topic.get("start_minute", 0),
            topic.get("topic_id", 0)
        )
    )
    filtered = ranked_topics[:max_topics]

    chronological_topics = sorted(
        filtered,
        key=lambda topic: (
            topic.get("start_minute", 0),
            topic.get("end_minute", 0),
            topic.get("topic_id", 0)
        )
    )

    llm_topics = []
    for i, topic in enumerate(chronological_topics):
        evidence_budget = _evidence_budget(topic["time_minutes"])
        evidence = topic.get("representative_evidence", [])[:evidence_budget]
        llm_topics.append({
            "order": i + 1,
            "time_minutes": round(topic["time_minutes"], 1),
            "time_range": topic.get("time_range"),
            "start_minute": topic.get("start_minute"),
            "end_minute": topic.get("end_minute"),
            "topic_type": topic.get("topic_type"),
            "topic_hint": topic.get("topic_hint"),
            "speaker_count": topic.get("speaker_count"),
            "dominant_speaker_ratio": topic.get("dominant_speaker_ratio"),
            "keywords": topic.get("top_lemmas", [])[:6],
            "segments_count": topic.get("segments_count"),
            "discussion_intensity": topic.get("discussion_intensity"),
            "topic_summary_hint": topic.get("topic_summary_hint"),
            "evidence": evidence,
        })

    total_minutes = round(sum(topic["time_minutes"] for topic in llm_topics), 1)
    procedural_minutes = round(
        sum(topic["time_minutes"] for topic in llm_topics if topic["topic_type"] == "procedural"),
        1
    )
    discussion_minutes = round(
        sum(
            topic["time_minutes"]
            for topic in llm_topics
            if topic["topic_type"] == "discussion" or topic["discussion_intensity"] == "high"
        ),
        1
    )

    longest_topics = sorted(
        llm_topics,
        key=lambda topic: (-topic["time_minutes"], topic["order"])
    )[:max_priority_topics]

    dominant_share = round(
        longest_topics[0]["time_minutes"] / total_minutes,
        2
    ) if total_minutes and longest_topics else 0.0

    if dominant_share >= 0.45:
        meeting_character = "dominated_by_one_topic"
    elif len(llm_topics) >= 6 and dominant_share <= 0.25:
        meeting_character = "spread_across_many_topics"
    else:
        meeting_character = "mixed_focus"

    return {
        "meeting_overview": {
            "total_meeting_minutes": total_minutes,
            "included_topic_count": len(llm_topics),
            "top_3_longest_topics": [
                {
                    "order": topic["order"],
                    "topic_hint": topic["topic_hint"],
                    "time_minutes": topic["time_minutes"],
                    "time_range": topic["time_range"],
                    "discussion_intensity": topic["discussion_intensity"],
                }
                for topic in longest_topics
            ],
            "procedural_share": round((procedural_minutes / total_minutes), 2) if total_minutes else 0.0,
            "discussion_share": round((discussion_minutes / total_minutes), 2) if total_minutes else 0.0,
            "meeting_character": meeting_character,
            "dominant_topic_share": dominant_share,
        },
        "priority_topics": [
            {
                "order": topic["order"],
                "topic_hint": topic["topic_hint"],
                "time_minutes": topic["time_minutes"],
                "time_range": topic["time_range"],
                "topic_type": topic["topic_type"],
                "discussion_intensity": topic["discussion_intensity"],
                "topic_summary_hint": topic["topic_summary_hint"],
            }
            for topic in longest_topics
        ],
        "topics": llm_topics,
    }


# =========================
# MAIN
# =========================

def main():
    """
    Main entry point for the meeting analyzer.

    Runs the complete deterministic pipeline:
    1. Loads and parses transcript file
    2. Merges consecutive utterances from same speaker
    3. Divides into overlapping time segments
    4. Lemmatizes text and builds TF-IDF vectors
    5. Clusters segments into topics using HDBSCAN
    6. Extracts topic summaries with representative sentences
    7. Prepares structured LLM input

    Outputs:
        - segments.json: All time-based segments with speaker info
        - topics.json: Full topic analysis with metadata
        - llm_input.json: Filtered and formatted topics for LLM

    Command line arguments:
        --file: Path to transcript file (required)
        --outdir, -o: Output directory (default: ./out)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", 
                        type=Path,
                        help="Transcription which should be used. Required.")
    parser.add_argument("--outdir",
                        "-o",
                        type=Path,
                        default=Path("out"),
                        help="Output directory (default: ./out)"
                    )

    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(f"{args.file} is not a valid file")

    try:
        args.outdir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError("Cannot create output directory") from e
    
    print("Loading transcript...")
    df = load_transcript(args.file)

    print("Merging utterances...")
    df = merge_utterances(df)

    print("Building segments...")
    segments = build_segments(df, args.outdir)

    print("Lemmatizing + TF-IDF...")
    lemmatizer = Lemmatizer(MORPHODITA_MODEL)
    X, vectorizer, lemma_maps = build_tfidf(segments, lemmatizer)

    print("Clustering...")
    labels = cluster_segments(X)

    print("Summarizing topics...")
    topics = summarize_topics(segments, labels, lemma_maps)

    print("\n=== TOPICS ===")
    for t in topics:
        print(
            f"\nTopic {t['topic_id']} | {t['time_minutes']:.1f} min"
        )
        print(", ".join(t["top_lemmas"]))

    Path(args.outdir / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    llm_payload = build_llm_query_payload(
        topics,
        min_minutes=3.0,
        max_topics=10
    )

    Path(args.outdir / "llm_input.json").write_text(
        json.dumps(llm_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
