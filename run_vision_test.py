import argparse
from pathlib import Path
from sensenova_eval.backend import MLXBackend


def main():
    parser = argparse.ArgumentParser(description="Run an image understanding smoke test")
    parser.add_argument("image", help="Path to a local image")
    parser.add_argument("--model", default="nativ-community/SenseNova-Loop-9B-4bit")
    parser.add_argument("--prompt", default="Describe this image accurately. Mention the main objects, colors, and spatial relationships.")
    args = parser.parse_args()
    if not Path(args.image).is_file():
        parser.error(f"Image not found: {args.image}")
    backend = MLXBackend(args.model)
    print(backend.vision(args.image, args.prompt, max_tokens=256).text)


if __name__ == "__main__":
    main()
