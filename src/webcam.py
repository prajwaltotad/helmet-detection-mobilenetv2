import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

try:
    from config import MODEL_PATH, PERSON_MODEL_PATH, IMG_SIZE
except ImportError:
    from src.config import MODEL_PATH, PERSON_MODEL_PATH, IMG_SIZE


def main():
    print("Loading helmet classifier...")
    helmet_model = load_model(MODEL_PATH, compile=False)

    print("Loading person detector...")
    person_model = YOLO(PERSON_MODEL_PATH)

    print("Models loaded successfully!")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read webcam frame.")
            break

        # Detect ONLY people (COCO class 0 = person)
        results = person_model(
            frame,
            classes=[0],
            conf=0.45,
            verbose=False
        )

        person_found = False

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                person_conf = float(box.conf[0])

                # Keep coordinates inside frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                person_found = True

                # Take only the upper portion of the person's bounding box.
                # This focuses on the head/helmet area.
                person_height = y2 - y1
                upper_y2 = y1 + int(person_height * 0.45)

                head_region = frame[y1:upper_y2, x1:x2]

                if head_region.size == 0:
                    continue

                # Prepare image for MobileNetV2
                img = cv2.resize(head_region, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=0)

                # IMPORTANT:
                # Do NOT call preprocess_input().
                # The trained model already contains MobileNetV2 preprocessing.

                prediction = float(
                    helmet_model.predict(img, verbose=0)[0][0]
                )

                if prediction >= 0.5:
                    label = "WITHOUT HELMET"
                    confidence = prediction
                    color = (0, 0, 255)
                else:
                    label = "WITH HELMET"
                    confidence = 1 - prediction
                    color = (0, 255, 0)

                # Draw person bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw label
                text = f"{label}: {confidence * 100:.1f}%"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                # Show the area actually sent to MobileNetV2
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, upper_y2),
                    (255, 255, 0),
                    1
                )

        if not person_found:
            cv2.putText(
                frame,
                "NO PERSON DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        cv2.imshow(
            "Helmet Detection - Press 'q' to quit",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()