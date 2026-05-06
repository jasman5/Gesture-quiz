import cv2, json, time, torch
import numpy as np
from models.gesture_net import GestureNet

# ---------------- MODEL ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureNet(num_classes=5)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

gesture_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "None"}

# ---------------- LOAD QUESTIONS ----------------
with open("data/questions.json") as f:
    quiz = json.load(f)

cap = cv2.VideoCapture(0)

score = 0
topic_scores = {t: [0, 0] for t in quiz.keys()}

# ---------------- PREPROCESS ----------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img

# ---------------- TEXT FUNCTION ----------------
def draw_text_with_bg(img, text, x, y, max_width,
                      line_height=30,
                      font=cv2.FONT_HERSHEY_SIMPLEX,
                      scale=0.7,
                      text_color=(0,0,0),        # BLACK TEXT
                      bg_color=(255,255,255),    # WHITE BG
                      thickness=2):

    words = text.split(' ')
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)

        if w > max_width:
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line = test_line

    lines.append(current_line)

    for i, line in enumerate(lines):
        (w, h), _ = cv2.getTextSize(line, font, scale, thickness)

        # Background rectangle
        cv2.rectangle(img,
                      (x-5, y + i*line_height - h - 5),
                      (x + w + 5, y + i*line_height + 5),
                      bg_color,
                      -1)

        # Text
        cv2.putText(img, line,
                    (x, y + i*line_height),
                    font, scale, text_color, thickness)

# ---------------- QUIZ LOOP ----------------
for topic, questions in quiz.items():
    for q in questions:
        start = time.time()
        selected = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = preprocess(frame).to(device)
            with torch.no_grad():
                preds = model(tensor)
                pred_idx = preds.argmax(1).item()
                gesture = gesture_map[pred_idx]

            h, w, _ = frame.shape
           # ---------------- GESTURE GUIDE (RHS) ----------------
            guide_x = int(w * 0.68)
            guide_y = 90

            # Smaller background box
            cv2.rectangle(frame, (guide_x-10, guide_y-30),
              (guide_x+260, guide_y+170), (255,255,255), -1)

            # Title
            cv2.putText(frame, "GESTURES", (guide_x, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            # Compact text
            cv2.putText(frame, "A -> 1 Finger", (guide_x, guide_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            cv2.putText(frame, "B -> Peace sign (2 fingers)", (guide_x, guide_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            cv2.putText(frame, "C -> C Shape hand", (guide_x, guide_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            cv2.putText(frame, "D -> Yo sign (pinky and first finger only)", (guide_x, guide_y + 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

            cv2.putText(frame, "None -> Back hand", (guide_x, guide_y + 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1)

            # Topic (Yellow)
            cv2.putText(frame, f"Topic: {topic}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Question (BLACK TEXT + WHITE BG)
            draw_text_with_bg(frame,
                              q["q"],
                              30, 80,
                              max_width=int(w * 0.6))

            # Options
            y_offset = 170
            for i, opt in enumerate(q["options"]):
                option_letter = chr(65+i)

                # CYAN highlight if selected
                if option_letter == gesture:
                    bg = (255,255,0)   # CYAN
                else:
                    bg = (255,255,255) # WHITE

                draw_text_with_bg(frame,
                                  f"{option_letter}. {opt}",
                                  50,
                                  y_offset + i*80,
                                  max_width=int(w * 0.55),
                                  text_color=(0,0,0),   # BLACK
                                  bg_color=bg)

            # Gesture (Green)
            cv2.putText(frame, f"Gesture: {gesture}",
                        (w-300, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("AI Quiz", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if time.time() - start > 10:
                selected = gesture
                break

        # Scoring
        correct = q["ans"]
        topic_scores[topic][1] += 1

        if selected == correct:
            score += 1
            topic_scores[topic][0] += 1

cap.release()
cv2.destroyAllWindows()

# ---------------- RESULTS ----------------
print("\n=== RESULTS ===")
for t, (c, total) in topic_scores.items():
    acc = (c / total) * 100 if total else 0
    print(f"{t}: {acc:.1f}%")

weak_topic = min(topic_scores,
                 key=lambda x: topic_scores[x][0] / max(1, topic_scores[x][1]))

print(f"Weak topic: {weak_topic}")
print(f"Total Score: {score}")