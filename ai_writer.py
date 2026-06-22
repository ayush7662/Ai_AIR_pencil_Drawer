import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------
# MediaPipe Setup
# -------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


# -------------------
# Streamlit UI
# -------------------

st.set_page_config(
    page_title="AI Air Pencil",
    layout="wide"
)

st.title("✋ AI Air Pencil Drawer")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])


# -------------------
# Camera
# -------------------

if run:

    cap = cv2.VideoCapture(0)

    canvas = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    prev_x, prev_y = 0, 0


    while run:

        success, frame = cap.read()

        if not success:
            st.error("Camera not found")
            break


        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        result = hand_landmarker.detect(mp_image)



        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:


                # index finger tip
                index_tip = hand_landmarks[8]


                x = int(index_tip.x*w)
                y = int(index_tip.y*h)


                cv2.circle(
                    frame,
                    (x,y),
                    10,
                    (0,255,0),
                    -1
                )


                if prev_x == 0:
                    prev_x,prev_y = x,y



                # draw air line

                cv2.line(
                    canvas,
                    (prev_x,prev_y),
                    (x,y),
                    (255,0,255),
                    5
                )


                prev_x,prev_y=x,y



                # landmarks

                for lm in hand_landmarks:

                    lx=int(lm.x*w)
                    ly=int(lm.y*h)

                    cv2.circle(
                        frame,
                        (lx,ly),
                        3,
                        (0,255,255),
                        -1
                    )


        else:
            prev_x,prev_y=0,0



        # merge canvas

        gray = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        _,mask=cv2.threshold(
            gray,
            20,
            255,
            cv2.THRESH_BINARY
        )


        mask_inv=cv2.bitwise_not(mask)


        bg=cv2.bitwise_and(
            frame,
            frame,
            mask=mask_inv
        )


        fg=cv2.bitwise_and(
            canvas,
            canvas,
            mask=mask
        )


        output=cv2.add(bg,fg)



        FRAME_WINDOW.image(
            cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            )
        )


    cap.release()


else:
    st.info("Click Start Camera")