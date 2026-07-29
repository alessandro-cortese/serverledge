import hashlib
import os


def handler(params, context):
    megabytes = int(params.get("megabytes", 32))

    path = "/tmp/serverledge_profiling_io_{}.bin".format(
        os.getpid(),
    )

    chunk = b"x" * (1024 * 1024)

    try:
        with open(path, "wb", buffering=0) as output_file:
            for _ in range(megabytes):
                output_file.write(chunk)

            os.fsync(output_file.fileno())

        digest = hashlib.sha256()

        with open(path, "rb", buffering=0) as input_file:
            while True:
                data = input_file.read(1024 * 1024)

                if not data:
                    break

                digest.update(data)

        return "processed_mb={}, sha256={}".format(
            megabytes,
            digest.hexdigest(),
        )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass