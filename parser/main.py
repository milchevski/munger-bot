"""
Entry point for the Berkshire Hathaway transcript parser.

Parses Transcripts.pdf and saves one JSON file per year into output/.
"""

from transcript_parser import parse_transcripts


def main():
    parse_transcripts("Transcripts.pdf", output_dir="documents")


if __name__ == "__main__":
    main()
