from pathlib import Path


def write_result(speed):
    base_folder = Path(__file__).parent.resolve()

    estimate_kmps_formatted = "{:.4f}".format(speed)

    output_string = estimate_kmps_formatted

    file_path = file_path = base_folder / "result.txt"
    with open(file_path, "w") as file:
        file.write(output_string)
