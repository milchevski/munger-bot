"""
Berkshire Hathaway Annual Meeting Transcript Parser.

Reads Transcripts.pdf and returns a dictionary mapping question titles
to [Buffett answer, Munger answer] pairs. Only includes Q&A pairs where
both Buffett and Munger respond.

Uses pypdfium2 for lightweight PDF text extraction to avoid OOM.
"""

import json
import os
import re
import pypdfium2 as pdfium


# Speaker label pattern
_SPEAKER_PATTERN = re.compile(
    r'^([A-Z][A-Z\s\.]+?):\s*',
    re.MULTILINE,
)

# Year header pattern
_YEAR_HEADER_PATTERN = re.compile(
    r'(?:Morning Session|Afternoon Session)\s*[-–—]\s*(\d{4})\s+Meeting'
)

# Pattern to strip zone chatter from the start of Buffett's answers
# Matches: "Zone 1?", "OK, Zone 2.", "Let's go to zone 3.", etc.
_ZONE_CHATTER_PATTERN = re.compile(
    r'^(?:OK[,.\s]*)?'
    r'(?:(?:Let\'s|let\'s)\s+(?:go|move)\s+(?:to|on\s+to)\s+)?'
    r'(?:[Zz]one|[Ss]tation|[Aa]rea)\s*\d+[A-Za-z]?[.,?\s]*'
    r'(?:\n+)?',
)

# Pattern to strip question-number chatter from the start of Buffett's answers
# Matches: "Number 8.", "OK Number 10.", "OK, number 3.", etc.
_NUMBER_CHATTER_PATTERN = re.compile(
    r'^(?:OK[,.\s]*)?'
    r'[Nn]umber\s+\d+[.?]?\s*',
)


def _parse_speaker_turns(text: str) -> list[tuple[str, str]]:
    """Parse text into (speaker, speech_text) tuples."""
    matches = list(_SPEAKER_PATTERN.finditer(text))
    if not matches:
        return []

    turns = []
    for idx, match in enumerate(matches):
        speaker = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        speech = text[start:end].strip()
        turns.append((speaker, speech))
    return turns


def _extract_qa(section_text: str) -> tuple[str, str] | None:
    """
    Extract Buffett's and Munger's answers from a Q&A section.

    Returns (buffett_answer, munger_answer) or None if:
    - No AUDIENCE MEMBER question is found
    - Buffett doesn't answer
    - Munger doesn't answer
    """
    turns = _parse_speaker_turns(section_text)
    if not turns:
        return None

    has_audience = any(
        speaker.startswith("AUDIENCE MEMBER") for speaker, _ in turns
    )
    if not has_audience:
        return None

    buffett_parts = []
    munger_parts = []
    for speaker, speech in turns:
        if speaker == "WARREN BUFFETT":
            # Strip zone/number chatter from the start of Buffett's speech
            speech = _ZONE_CHATTER_PATTERN.sub('', speech)
            speech = _NUMBER_CHATTER_PATTERN.sub('', speech).strip()
            if speech:
                buffett_parts.append(speech)
        elif speaker == "CHARLIE MUNGER":
            munger_parts.append(speech)

    if not buffett_parts or not munger_parts:
        return None

    buffett_answer = " ".join(buffett_parts)
    munger_answer = " ".join(munger_parts)
    # Normalize all whitespace: \r\n, \n, multiple spaces -> single space
    buffett_answer = re.sub(r'[\r\n]+', ' ', buffett_answer)
    buffett_answer = re.sub(r'\s{2,}', ' ', buffett_answer).strip()
    munger_answer = re.sub(r'[\r\n]+', ' ', munger_answer)
    munger_answer = re.sub(r'\s{2,}', ' ', munger_answer).strip()
    return (buffett_answer, munger_answer)


def _save_year(output_dir: str, year: str, data: dict) -> None:
    """Save a year's Q&A pairs to a JSON file."""
    if not data:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"transcripts_{year}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}: {len(data)} Q&A pairs")


def parse_transcripts(file_path: str, output_dir: str = "documents") -> dict:
    """
    Parse Berkshire Hathaway annual meeting transcripts from a PDF file.

    Processes pages one at a time using pypdfium2 (low memory footprint).
    Writes a separate JSON file per year to the output directory.

    Args:
        file_path: Path to the Transcripts.pdf file.
        output_dir: Directory to write per-year JSON files.

    Returns:
        Summary dict with stats about what was parsed.
    """
    pdf = pdfium.PdfDocument(file_path)
    total = len(pdf)
    print(f"Parsing {total} pages...")

    current_year = "unknown"
    current_section_title: str | None = None
    current_section_lines: list[str] = []

    year_data: dict[str, list[str]] = {}  # current year's Q&A pairs
    total_qa = 0
    total_sections = 0
    skipped_no_qa = 0
    skipped_no_munger = 0
    years_saved: list[str] = []

    def _flush_section():
        nonlocal total_sections, skipped_no_qa, skipped_no_munger
        if current_section_title and current_section_lines:
            total_sections += 1
            text = "\n".join(current_section_lines)
            qa = _extract_qa(text)
            if qa is None:
                turns = _parse_speaker_turns(text)
                has_audience = any(
                    s.startswith("AUDIENCE MEMBER") for s, _ in turns
                )
                if not has_audience:
                    skipped_no_qa += 1
                else:
                    skipped_no_munger += 1
                return
            key = current_section_title
            if key in year_data:
                counter = 2
                while f"{key} ({counter})" in year_data:
                    counter += 1
                key = f"{key} ({counter})"
            year_data[key] = list(qa)

    def _flush_year():
        nonlocal total_qa, year_data
        if year_data:
            total_qa += len(year_data)
            _save_year(output_dir, current_year, year_data)
            years_saved.append(current_year)
        year_data = {}

    for i in range(total):
        page = pdf[i]
        textpage = page.get_textpage()
        page_text = textpage.get_text_range()
        textpage.close()
        page.close()

        if not page_text:
            continue

        lines = page_text.split("\n")

        for line in lines:
            # Check for year header
            year_match = _YEAR_HEADER_PATTERN.search(line)
            if year_match:
                _flush_section()
                new_year = year_match.group(1)
                if new_year != current_year:
                    _flush_year()
                    current_year = new_year
                current_section_title = None
                current_section_lines = []
                continue

            # Check for numbered section header
            section_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if section_match:
                _flush_section()
                current_section_title = section_match.group(2).strip()
                current_section_lines = []
                continue

            # Accumulate text for current section
            if current_section_title is not None:
                current_section_lines.append(line)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total} pages "
                  f"({total_qa + len(year_data)} Q&A pairs so far)...")

    # Flush remaining
    _flush_section()
    _flush_year()

    pdf.close()

    summary = {
        "total_sections": total_sections,
        "total_qa": total_qa,
        "skipped_no_qa": skipped_no_qa,
        "skipped_no_munger": skipped_no_munger,
        "years": sorted(years_saved),
    }

    print(f"\nParsing complete:")
    print(f"  Total sections: {summary['total_sections']}")
    print(f"  Q&A pairs with both Buffett & Munger: {summary['total_qa']}")
    print(f"  Skipped (not Q&A): {summary['skipped_no_qa']}")
    print(f"  Skipped (missing Munger): {summary['skipped_no_munger']}")
    print(f"  Years: {summary['years']}")

    return summary
