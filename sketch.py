import cv2
import numpy as np
import mediapipe as mp
import math


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    stream = cv2.VideoCapture(0)
    ret, frame = stream.read()
    h, w, _ = frame.shape
    sketch = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # create the color strip, which is in two parts (color and black/white)
    strip_h = int(h*0.04)
    strip_w = int(w*0.7)
    middle = strip_w*5//7
    color_strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    for x in range(middle):
        hue = 179-int((x/middle)*179)
        hue -= middle//40
        if hue < 0: hue += 179
        color_strip[:, x] = (hue, 255, 255)
    color_strip = cv2.cvtColor(color_strip, cv2.COLOR_HSV2BGR)
    for x in range(middle, strip_w):
        shade = int(((x-middle)/(strip_w-middle))*255)
        color_strip[:, x] = (shade, shade, shade)
    eraser_line = middle//16
    color_strip[:, 0:eraser_line] = (255, 255, 255)
    cv2.rectangle(color_strip, (0, 0), (eraser_line, strip_h-1), (0, 0, 0), 1)
    cv2.line(color_strip, (0, 0), (eraser_line, strip_h-1), (0, 0, 0), 1)
    cv2.line(color_strip, (eraser_line, 0), (0, strip_h-1), (0, 0, 0), 1)

    # the distance from the side the color bar starts
    color_side = (w-strip_w)//2
    eraser_line += color_side
    # the color slider's position
    color_slider = middle + color_side

    # create the thickness bar and slider
    bar_h = int(h*0.8)
    bar_w = int(w*0.04)
    thickness_bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    thickness_bar[:, :] = (255, 255, 255)
    top = (h-bar_h)//2
    max_thickness = 100
    thickness_slider = int(bar_h/5+top)

    # Initialize color and thickness
    color = (0, 0, 0)
    thickness = int((thickness_slider-top)/bar_h*max_thickness)

    # maybe print a menu/instructions?

    while True:
        ret, frame = stream.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(complement(color) if color else (0, 0, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color if color else (255, 255, 255), thickness=2)
                )
                
                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                knuckle1 = hand_landmarks.landmark[6]
                knuckle2 = hand_landmarks.landmark[7]
                x = int(index_tip.x*w)
                y = int(index_tip.y*h)
                threshold = max(0.05, distance(index_tip, knuckle1, w, h))
                pinch_distance = distance(index_tip, thumb_tip, w, h)
                # If fingers are pinching and close to one of the sliders
                if pinch_distance < distance(index_tip, knuckle2, w, h) and index_tip.x*w > w-(bar_w*3):
                    center = int((index_tip.y+thumb_tip.y)*h//2)
                    if top < center < top+bar_h:
                        thickness_slider = center
                        thickness = max(1, int((thickness_slider-top)/bar_h*max_thickness))
                        cv2.circle(frame, (x, y), thickness//2, color if color else (255, 255, 255), thickness=-1)
                        cv2.circle(frame, (x, y), thickness//2, complement(color) if color else (0, 0, 0), thickness=1)
                elif pinch_distance < distance(index_tip, knuckle2, w, h) and index_tip.y*h < strip_h*3:
                    center = int((index_tip.x+thumb_tip.x)*w//2)
                    if color_side < center < color_side+strip_w:
                        color_slider = center
                        # if slider is in eraser
                        if color_slider < eraser_line: color = None
                        else:
                            color = color_strip[0, center-color_side]
                            color = (int(color[0]), int(color[1]), int(color[2]))
                    thickness_bar[:, :] = color if color else (255, 255, 255)

                # Check if the index tip is too close to another hand part
                for i in range(21):
                    # Certain hand parts can be close to the index tip
                    if i in [6, 7, 8, 10]: continue
                    curr_point = hand_landmarks.landmark[i]
                    d = distance(index_tip, curr_point, w, h)
                    if abs(d) < threshold: break
                else:
                    # draw on the sketch
                    draw_color = color
                    mask_color = 1
                    if not color:
                        mask_color = 0
                        draw_color = (255, 255, 255)
                    if last_index_position:
                        cv2.line(sketch, (last_index_position[0], last_index_position[1]), (x, y), draw_color, thickness, cv2.LINE_AA)
                        cv2.line(mask, (last_index_position[0], last_index_position[1]), (x, y), mask_color, thickness, cv2.LINE_8)
                    else:
                        cv2.circle(sketch, (x, y), thickness//2, draw_color, thickness=-1)
                        cv2.circle(mask, (x, y), thickness//2, mask_color, thickness=-1)
                    cv2.circle(frame, (x, y), thickness//2, draw_color, thickness=-1)
                    cv2.circle(frame, (x, y), thickness//2, complement(draw_color), thickness=1)
            last_index_position = (x, y)

        # No hands in frame
        else: last_index_position = None
                
        # Put drawings and color/thickness objects on the frame
        frame[mask.astype(bool)] = sketch[mask.astype(bool)]
        frame[0:strip_h, color_side:w-color_side] = color_strip
        #color slider
        cv2.line(frame, (color_slider, 0), (color_slider, strip_h*2), (0, 0, 0), 3)
        # thickness bar and slider
        frame[top:bar_h+top, w-bar_w:w] = thickness_bar
        cv2.line(frame, (w, thickness_slider), (w-bar_w*2, thickness_slider), (0, 0, 0), 3)

        cv2.imshow('Sketch', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == 27: break

    stream.release()
    cv2.destroyAllWindows()

def distance(p1, p2, w, h):
    """ Uses Pythagora's theorem to get the distance between two points """
    x1 = p1.x*w
    y1 = p1.y*h
    x2 = p2.x*w
    y2 = p2.y*h
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

complement = lambda color: (255-color[0], 255-color[1], 255-color[2])

if __name__ == "__main__":
    main()