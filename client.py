import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 3.5 Turbo client")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--host", type=str, default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image filename")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/generate"
    print(f"➡️ Sending request to {api_url} ...")

    response = requests.get(api_url, params={"prompt": args.prompt})
    if response.status_code == 200:
        with open(args.output, "wb") as f:
            f.write(response.content)
        print(f"✅ Image saved as {args.output}")
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    main()
