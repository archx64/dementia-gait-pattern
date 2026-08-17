import yaml

INPUT_YAML = "edit_frames/mahidol.yaml"
OUTPUT_YAML = "edit_frames/mahidol_updated.yaml"


def process_yaml(input_file, output_file):
    try:
        # load the YAML data
        with open(input_file, "r") as file:
            # Note: Ensure the missing quote in r28 ('11:10:29) is fixed in the
            # original file, otherwise safe_load will throw a parsing error!
            data = yaml.safe_load(file)

        # 2. Navigate the specific nested structure
        # Structure: data['01_April']['p1']['rX']['c1']
        if "01_April" in data and "p1" in data["01_April"]:
            rounds = data["01_April"]["p1"]

            for round_name, round_data in rounds.items():
                if "c1" in round_data:
                    c1_data = round_data["c1"]

                    start_frame = c1_data.get("start_frame")
                    end_frame = c1_data.get("end_frame")

                    # 3. Calculate and inject total_frames if both values exist
                    if start_frame is not None and end_frame is not None:
                        c1_data["total_frames"] = end_frame - start_frame

        # 4. Save the reformatted YAML
        with open(output_file, "w") as file:
            yaml.dump(
                data,
                file,
                default_flow_style=False,  # Keeps the block format instead of inline dicts
                sort_keys=False,  # Preserves the original top-to-bottom order
                indent=2,  # Enforces strict 2-space horizontal spacing
            )

        print(f"Successfully calculated frames and saved to {output_file}")

    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file. Please check for syntax errors: {e}")
    except FileNotFoundError:
        print(f"Could not find the file named {input_file}")


if __name__ == "__main__":
    process_yaml(INPUT_YAML, OUTPUT_YAML)
