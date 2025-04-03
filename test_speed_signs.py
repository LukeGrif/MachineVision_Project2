import os
import re
import matplotlib
matplotlib.use('Agg')
from SignSelector import SpeedSignDetector

# Folder containing your test images
TEST_IMAGES_DIR = r"speed-sign-test-images"

def parse_expected_from_filename(filename):
    """
    Parses the filename of the format '50-0004x2.png' to extract:
      - Expected speed (50) as an integer.
      - Expected count (2) of signs.
    """
    pattern = r"(\d+)-\d+x(\d+)\.png"
    match = re.match(pattern, filename)
    if match:
        expected_speed = int(match.group(1))
        expected_count = int(match.group(2))
        return expected_speed, expected_count
    return None, None

def run_tests():
    detector = SpeedSignDetector()
    total_tests = 0
    passed = 0
    for file in os.listdir(TEST_IMAGES_DIR):
        if file.endswith(".png"):
            total_tests += 1
            filepath = os.path.join(TEST_IMAGES_DIR, file)
            expected_speed, expected_count = parse_expected_from_filename(file)
            if expected_speed is None:
                print(f"Skipping file with unexpected format: {file}")
                continue

            results = detector.process_image(filepath)
            detected_count = len(results)
            speeds = [r['speed'] for r in results]
            
            # Determine if the detected results match the expected count and speed
            if detected_count == expected_count and all(speed == expected_speed for speed in speeds):
                print(f"[PASS] {file}: Detected {detected_count} sign(s) with speed {expected_speed}.")
                passed += 1
            else:
                print(f"[FAIL] {file}: Expected {expected_count} sign(s) with speed {expected_speed} but detected {detected_count} sign(s) with speeds {speeds}.")
    print(f"Passed {passed} out of {total_tests} tests.")

if __name__ == "__main__":
    run_tests()