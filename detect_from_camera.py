import numpy as np
import cv2
import torch
from models import Resnet18_gray


def detect_keypoints(img_size, save=False):

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
    model = Resnet18_gray()
    state_dict = torch.load(
        "saved_models/keypoints_resnet18_final.pt", map_location=torch.device("cpu"),
    )
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
            gray, scaleFactor=1.15, minNeighbors=3, minSize=(75, 75),
        )

        # display the resulting frame
        for (x, y, w, h) in faces:
            # select the region of interest that is the face in the image
            roi = gray[y : y + h, x : x + w]

            ## normalize, rescale and reshape
            shape_before_resize = roi.shape
            roi_rescaled = cv2.resize(roi, (img_size, img_size))
            roi_normalized = (
                roi_rescaled / 255.0 / 255.0
            )  # due to mpimg.imread(image_name) where reading training data
            shape_after_resize = roi_normalized.shape
            scaling_factor_y = shape_before_resize[0] / shape_after_resize[0]
            scaling_factor_x = shape_before_resize[1] / shape_after_resize[1]
            roi_torch = torch.from_numpy(
                roi_normalized.reshape((1, 1, img_size, img_size))
            ).float()

            ## predict
            pred_keypoints = model(roi_torch)
            keypoints_np = pred_keypoints.view((68, -1)).data.numpy()
            keypoints_unnorm = keypoints_np * (roi_normalized.shape[0] / 4) + (
                roi_normalized.shape[0] / 2
            )
            keypoints_list = [
                (
                    int(np.round(ptx * scaling_factor_x, 0) + x),  # - x_minus),
                    int(np.round(pty * scaling_factor_y, 0) + y),  # - y_minus),
                )
                for ptx, pty in zip(keypoints_unnorm[:, 0], keypoints_unnorm[:, 1])
            ]

            # add keypoints to frame
            for i in range(len(keypoints_list)):

                if i in [16, 21, 26, 35]:
                    continue

                prev = keypoints_list[i]
                if i not in [30, 41, 47, 59, 67]:
                    next = keypoints_list[i + 1]
                elif i == 30:
                    next = keypoints_list[33]
                elif i == 41:
                    next = keypoints_list[36]
                elif i == 47:
                    next = keypoints_list[42]
                elif i == 59:
                    next = keypoints_list[48]
                elif i == 67:
                    next = keypoints_list[60]

                # colors
                if i < 16 or (i > 25 and i < 35):
                    color = (85, 222, 250)  # contour and nose
                elif i < 26:
                    color = (255, 89, 252)  # eyebrows
                elif i < 48:
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
    detect_keypoints(img_size=224, save=True)
