import argparse
import json
from pathlib import Path

from orbitdesk_agent.generation import LocalGenerator
from orbitdesk_agent.retrieval import LocalRetriever
from orbitdesk_agent.workflow import build_workflow


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local OrbitDesk support agent.")
    parser.add_argument("query", help="Support question to answer.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--offline", action="store_true", help="Use only already downloaded Hugging Face models.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--generation-revision", default="main")
    parser.add_argument("--force-first-verification-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = LocalRetriever(
        args.data_dir,
        args.embedding_model,
        args.embedding_revision,
        args.offline,
    )
    generator = LocalGenerator(
        args.generation_model,
        args.generation_revision,
        args.offline,
    )
    workflow = build_workflow(retriever, generator)
    result = workflow.invoke(
        {
            "query": args.query,
            "retry_count": 0,
            "node_log": [],
            "force_first_verification_failure": args.force_first_verification_failure,
        }
    )
    response = {
        "classification": result["classification"],
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "requires_human": result["requires_human"],
        "reason": result["reason"],
        "clarification_question": result["clarification_question"],
        "warnings": result["warnings"],
    }
    print(json.dumps(response, indent=2))
    print(
        json.dumps(
            {
                "node_log": result["node_log"],
                "embedding_model": retriever.model_name,
                "embedding_revision": retriever.resolved_revision,
                "generation_model": generator.model_name,
                "generation_revision": generator.resolved_revision,
                "generation_load_seconds": generator.load_seconds,
                "generation_device": generator.device,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
