from SignSelector import SpeedSignDetector
import os
import re

detector = SpeedSignDetector()

images = []
pass_count = 0
fail_paths = []

for path in os.listdir("./speed-sign-test-images"):
    images.append("speed-sign-test-images\\\\" + path)

for path in images:
    print(f"Image {path}")

    results = detector.process_image(path)

    match = re.search("(\\d+)\\-00", path)

    if match is None:
        "could not extract speed from file name"
        fail_paths.append(path)
        print("\n")
        continue

    true_speed_limit = match.group(1)

    print(f"Actual speed: {true_speed_limit}")
    for result in results:
        passed = True
        print(f"Detect speed: {result['speed']}")
        passed = float(true_speed_limit) == float(result['speed'])
        print("PASS" if passed else "FAIL")
        if passed:
            pass_count += 1
        else:
            fail_paths.append(path)

    if len(results) == 0:
        print("No speed detected")
        print("FAIL")
        fail_paths.append(path)

    print("\n")

print("\nSUMMARY")
print(f"{pass_count} Passed")
print(f"{len(fail_paths)} Failed")
print("Images that failed:")
for path in fail_paths: print(path)


    # Display visual results
    # detector.display_results(image, results)

    # Print terminal output
    # detector.print_terminal_output(results)
