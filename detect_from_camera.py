import numpy as np
import cv2
import torch
from models import Net


def detect_keypoints(save=False):

    # capture image from camera
    cap = cv2.VideoCapture(-1)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    ## load models
    # face detector
    face_cascade = cv2.CascadeClassifier(
        "detector_architectures/haarcascade_frontalface_default.xml"
    )
    # keypoint prediction
    model = Net()
    state_dict = torch.load("saved_models/keypoints_model_adam_001_20ep_amsgrad.pt")
    model.load_state_dict(state_dict)
    model.eval()

    # output
    if save:
        out = cv2.VideoWriter(
            "output.avi",
            fourcc=cv2.VideoWriter_fourcc("F", "M", "P", "4"),
            fps=fps,
            frameSize=(width, height),
        )

    while True:
        # capture frame-by-frame
        if not cap.isOpened():
            cap.open()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        # change to gray and perform face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=3, minSize=(75, 75)  # 1.2, 2
        )

        # display the resulting frame
        for (x, y, w, h) in faces:
            # draw a rectangle around each detected face
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            # select the region of interest that is the face in the image
            d = 50
            y_minus = y if y < d else d
            y_plus = height - h - y if y + h + d > height else d
            x_minus = x if x < d else d
            x_plus = width - w - x if x + h + d > width else d
            roi = gray[y - y_minus : y + h + y_plus, x - x_minus : x + w + x_plus]

            ## normalize, rescale and reshape
            roi_normalized = roi / 255.0
            shape_before_resize = roi.shape
            roi_rescaled = cv2.resize(roi_normalized, (224, 224))
            shape_after_resize = roi_rescaled.shape
            scaling_factor_y = shape_before_resize[0] / shape_after_resize[0]
            scaling_factor_x = shape_before_resize[1] / shape_after_resize[1]
            roi_torch = torch.from_numpy(roi_rescaled.reshape((1, 1, 224, 224))).float()

            ## predict
            pred_keypoints = model(roi_torch)
            keypoints_np = pred_keypoints.view((68, -1)).data.numpy()
            keypoints_unnorm = keypoints_np * 50 + 100
            keypoints_list = [
                (
                    int(np.round(ptx * scaling_factor_x, 0) + x - x_minus),
                    int(np.round(pty * scaling_factor_y, 0) + y - y_minus),
                )
                for ptx, pty in zip(keypoints_unnorm[:, 0], keypoints_unnorm[:, 1])
            ]

            # add keypoints to frame
            for i in range(len(keypoints_list)):

                if i in [16, 21, 26, 35, 41, 47, 59, 67]:
                    continue

                prev = keypoints_list[i]
                next = keypoints_list[i + 1]

                # colors
                if i < 16 or (i > 25 and i < 35):
                    color = (85, 222, 250)  # contour and nose
                elif i < 26:
                    color = (33, 67, 79)  # eyebrows
                elif i < 47:
                    color = (245, 133, 34)  # eyes
                else:
                    color = (69, 69, 247)  # mouth

                # add
                cv2.line(
                    frame, prev, next, color=color, thickness=2,
                )

        # plot frame with keypoints
        cv2.imshow("frame", frame)

        # save to file
        if save:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # when everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_keypoints(save=True)
