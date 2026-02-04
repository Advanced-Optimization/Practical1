import asyncio
import struct
import torch
from modules.pytorch_mlp import PytorchMLPReg

regr = None

# 3 floats in, 4 floats out
INPUT_FMT = "!3f"     # network byte order
OUTPUT_FMT = "!4f"
INPUT_SIZE = struct.calcsize(INPUT_FMT)


async def handle_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter):
    global regr
    print("Client connected")
    try:
        while True:
                # Read exactly 3 floats
                data = await reader.readexactly(INPUT_SIZE)
                x, y, z = struct.unpack(INPUT_FMT, data)
                print(f"Received: {x}, {y}, {z}")

                # Prepare tensor
                target = torch.tensor(
                    [[x, y, z]],
                    dtype=torch.float32,
                    device="cpu",
                )

                with torch.inference_mode():
                    output = regr.predict(target)[0]
                    print("Infered:", output)

                output_bytes = struct.pack(
                    OUTPUT_FMT,
                    float(output[0]),
                    float(output[1]),
                    float(output[2]),
                    float(output[3]),
                )

                writer.write(output_bytes)
                await writer.drain()

    except asyncio.IncompleteReadError:
        print("Client disconnected early")

    except Exception as e:
        print("Server error:", e)

    finally:
        writer.close()
        await writer.wait_closed()


async def main(model_file):
    global regr
    regr = PytorchMLPReg(model_file=model_file, batch_size=1)
    server = await asyncio.start_server(
    handle_client, '127.0.0.1', 5000)
    async with server:
        print("serving on 127.0.0.1:5000")
        await server.serve_forever()
    print("Closed server")



if __name__ == "__main__":
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Simulate a leg.")
    parser.add_argument(
        metavar="model_file",
        type=str,
        nargs="?",
        help="the path to the file containing the model",
        dest="model_file",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        print(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    print(
        os.path.basename(__file__),
        f"Using model file: {os.path.join(os.path.dirname(os.path.realpath(__file__)), args.model_file)}",
    )
    
    asyncio.run(main(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.model_file)))

