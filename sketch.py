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
    _, frame = stream.read()
    h, w, _ = frame.shape
    sketch = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # create the color strip, which is in two parts (color and black/white)
    strip_w = int(w*0.7)
    strip_h = int(h*0.04)
    middle = strip_w*5//7
    eraser_line = middle//16
    color_strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    color_strip = create_color_strip(strip_w, strip_h, middle, eraser_line, color_strip)

    # the distance from the side the color bar starts
    color_side = (w-strip_w)//2
    eraser_line += color_side

    # the color slider's initial position
    color_slider = middle + color_side

    # create the thickness bar
    bar_h = int(h*0.8)
    bar_w = int(w*0.04)
    thickness_bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    thickness_bar[:, :] = (255, 255, 255)

    # the distance from the top the thickness bar starts
    top = (h-bar_h)//2

    # attributes for the slider
    max_thickness = 100
    thickness_slider = int(bar_h/5+top)

    # Initialize color and thickness
    color = (0, 0, 0)
    thickness = int((thickness_slider-top)/bar_h*max_thickness)
    last_index_position = None

    # print basic usage
    print_instructions()

    while True:
        ret, frame = stream.read()
        if not ret: break

        results = get_hands(frame)

        hand = False
        drawing = False
        if results.multi_hand_landmarks:
            hand = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            knuckle = hand_landmarks.landmark[6]
            x = int(index_tip.x*w)
            y = int(index_tip.y*h)
            threshold = max(0.07, distance(index_tip, knuckle, w, h))
            pinch_distance = distance(index_tip, thumb_tip, w, h)
            # If fingers are pinching and close to one of the sliders
            if pinch_distance < threshold and index_tip.x*w > w-(bar_w*3):
                center_x = int((index_tip.x+thumb_tip.x)*w//2)
                center_y = int((index_tip.y+thumb_tip.y)*h//2)
                if top < center_y < top+bar_h:
                    thickness_slider = center_y
                    thickness = max(1, int((thickness_slider-top)/bar_h*max_thickness))
                    cv2.circle(frame, (center_x, center_y), thickness//2, color if color else (255, 255, 255), thickness=-1)
                    cv2.circle(frame, (center_x, center_y), thickness//2, complement(color) if color else (0, 0, 0), thickness=1)

            elif pinch_distance < distance(index_tip, knuckle, w, h) and index_tip.y*h < strip_h*3:
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
                drawing = True
                draw_color = draw(color, thickness, (x, y), last_index_position, sketch, mask)
            last_index_position = (x, y)

        # No hands in frame
        else: last_index_position = None

        # layer things on frame in the correct order

        # Put drawings and color/thickness objects on the frame
        frame[mask.astype(bool)] = sketch[mask.astype(bool)]
        frame[0:strip_h, color_side:w-color_side] = color_strip

        # color slider that is twice the height of the strip
        cv2.line(frame, (color_slider, 0), (color_slider, strip_h*2), (0, 0, 0), 3)

        # thickness bar and slider that is twice the width of the strip
        frame[top:bar_h+top, w-bar_w:w] = thickness_bar
        cv2.line(frame, (w, thickness_slider), (w-bar_w*2, thickness_slider), (0, 0, 0), 3)

        # hand skeleton
        if hand:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(complement(color) if color else (0, 0, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(change_color(color) if color else (255, 255, 255), thickness=2)
            )
        # color dot
        if drawing:
            cv2.circle(frame, (x, y), thickness//2, draw_color, thickness=-1)
            cv2.circle(frame, (x, y), thickness//2, complement(draw_color), thickness=1)

        cv2.imshow('Sketch', cv2.flip(frame, 1))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'): save_background(frame, sketch, mask)
        elif key == ord('s'): save_sketch(sketch, mask, h, w)
        elif key == ord('c'): clear_sketch(sketch, mask)
        elif key == 27: break

    stream.release()
    cv2.destroyAllWindows()


def print_instructions():
    instructions = """
    How to use the virtual sketchpad:

    Draw with your index finger. Hold your hand as if to gesture the number '1'.
    Holding another finger close to the index finger will allow you to move your hand without drawing.

    You can drag the sliders along the top and left sides of the screen to change the color and line thickness.
    To do this, pinch the slider and move it. Make sure your hand stays inside the frame.

    The white 'X' on the right side of the color strip will turn your finger into an eraser.

    Make sure to keep your whole hand inside the frame!

    Once you're done drawing, here are your options:
        Press 'f' to save the image as a .png that includes your background.
        Press 's' to save the image as a .png that does not include your background.
        Press 'c' to clear the sketch.
        Press 'esc' to exit.
        Pressing 'f' or 's' will prompt you for an image name.
    """
    print(f"\n{instructions}\n")


def create_color_strip(strip_w, strip_h, middle, eraser_line, color_strip):
    # colored part
    for x in range(middle):
        hue = 179-int((x/middle)*179)
        # shifts so that the bar appears in rainbow order
        hue -= middle//40
        if hue < 0: hue += 179
        color_strip[:, x] = (hue, 255, 255)
    color_strip = cv2.cvtColor(color_strip, cv2.COLOR_HSV2BGR)
    # black/white part
    for x in range(middle, strip_w):
        shade = int(((x-middle)/(strip_w-middle))*255)
        color_strip[:, x] = (shade, shade, shade)
    # eraser
    color_strip[:, 0:eraser_line] = (255, 255, 255)
    cv2.rectangle(color_strip, (0, 0), (eraser_line, strip_h-1), (0, 0, 0), 1)
    cv2.line(color_strip, (0, 0), (eraser_line, strip_h-1), (0, 0, 0), 1)
    cv2.line(color_strip, (eraser_line, 0), (0, strip_h-1), (0, 0, 0), 1)
    return color_strip


def get_hands(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    return results


def draw(color, thickness, curr_position, last_index_position, sketch, mask):
    x = curr_position[0]
    y = curr_position[1]
    draw_color = color
    mask_color = 1
    if color is None:
        mask_color = 0
        draw_color = (255, 255, 255)
    if last_index_position:
        cv2.line(sketch, (last_index_position[0], last_index_position[1]), (x, y), draw_color, thickness, cv2.LINE_AA)
        cv2.line(mask, (last_index_position[0], last_index_position[1]), (x, y), mask_color, thickness, cv2.LINE_8)
    else:
        cv2.circle(sketch, (x, y), thickness//2, draw_color, thickness=-1)
        cv2.circle(mask, (x, y), thickness//2, mask_color, thickness=-1)
    return draw_color


def save_background(frame, sketch, mask):
    image = input("Image name ___.png (default is 'sketch.png'): ")
    if '.' in image: image = image[:image.find('.')]
    if not image.strip(): image = "sketch"
    image += ".png"
    frame[mask.astype(bool)] = sketch[mask.astype(bool)]
    status = cv2.imwrite(image, cv2.flip(frame, 1))
    print(f"Image saved to {image}: {status}")


def save_sketch(sketch, mask, h, w):
    image = input("Image name ___.png (default is 'sketch.png'): ").strip()
    if '.' in image: image = image[:image.find('.')]
    if not image: image = "sketch"
    image += ".png"
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = sketch
    rgba[:, :, 3] = (mask * 255)

    status = cv2.imwrite(image, cv2.flip(rgba, 1))
    print(f"Sketch saved to {image}: {status}")


def clear_sketch(sketch, mask):
    sketch[:, :] = 0
    mask[:, :] = 0


def distance(p1, p2, w, h):
    # Uses the Pythagorean Theorem to get the distance between two points
    x1 = p1.x*w
    y1 = p1.y*h
    x2 = p2.x*w
    y2 = p2.y*h
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


def change_color(color):
    r = color[0]-150
    g = color[1]-150
    b = color[2]-150
    if r < 0: r += 255
    if g < 0: g += 255
    if b < 0: b += 255
    return (r, g, b)


complement = lambda color: (255-color[0], 255-color[1], 255-color[2])

if __name__ == "__main__":
    main()