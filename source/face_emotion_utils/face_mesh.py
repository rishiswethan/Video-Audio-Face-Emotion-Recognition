import cv2 as cv2
import mediapipe as mp
from imutils import face_utils as imutils_face_utils
import PIL.Image as Image

import source.config as config
import source.face_emotion_utils.face_config as face_config


def get_mesh(image, upscale_landmarks=True, save_path=None, showImg=False, print_flag=True, return_mesh=False):
    """
    Get the face mesh from an image
    Parameters
    ----------
    image : np.ndarray
        The image to get the face mesh from. Must be BGR.

    Returns
    -------
    sub_face_landmark_list : list
        List of all the landmarks of the face
    sub_face_mini: np.ndarray
        The cropped face

    """

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=100, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        img_cpy = image.copy()
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(processed_img)

        if results.multi_face_landmarks is None:
            if print_flag:
                print('No mesh detected')
            return None
        elif len(results.multi_face_landmarks) > 1:
            print("More than one face detected:", len(results.multi_face_landmarks))
            return None

        annotated_image = image.copy()
        unannotated_image = image.copy()
        landmark_list = []
        depth_list = []
        for face_landmarks in results.multi_face_landmarks:
            # print('face_landmarks:', face_landmarks)

            # Append all landmarks to list
            min_y = min_y_real = 999999999999
            max_y = max_y_real = 0
            for i in range(468):
                x = face_landmarks.landmark[i].x
                y = face_landmarks.landmark[i].y
                z = face_landmarks.landmark[i].z

                x_real = int(x * image.shape[1])
                y_real = int(y * image.shape[0])

                if y < min_y:
                    min_y = y
                    min_y_real = y_real
                    min_x_real = x_real
                if y > max_y:
                    max_y = y
                    max_y_real = y_real
                    max_x_real = x_real

                landmark_list.append((x_real, y_real))
                depth_list.append(z)

            y_y_dist = max_y - min_y
            y_y_dist_real = max_y_real - min_y_real

            landmark_upscale_factor = 1 / y_y_dist
            depth_list.append(landmark_upscale_factor)  # Provide upscale factor info as well at the end of this list for NN to use

            top_left_x = max(min_x_real - int(y_y_dist_real / 2), 0)
            top_left_y = max(min_y_real, 0)
            w = h = y_y_dist_real

            # Get a mini sub_face to avoid the background
            decrease_bounding_box_per = -0.0
            x_small = max(int(top_left_x - decrease_bounding_box_per * w), 0)
            y_small = max(int(top_left_y - decrease_bounding_box_per * h), 0)
            w_small = int(w + decrease_bounding_box_per * w * 2)
            h_small = int(h + decrease_bounding_box_per * h * 2)

            # Move the bounding box to the left by 10% and 10% up
            # x_small = max(int(x_small - 0.15 * w_small), 0)
            # y_small = max(int(y_small - 0.15 * h_small), 0)

            sub_face_mini = img_cpy[y_small:y_small + h_small, x_small:x_small + w_small]
            # Convert it back to BGR
            sub_face_mini = cv2.cvtColor(sub_face_mini, cv2.COLOR_RGB2BGR)

            # Make bounding box slightly bigger to fit mesh
            increase_bounding_box_per = 0.2
            x_big = max(int(top_left_x - increase_bounding_box_per * w), 0)
            y_big = max(int(top_left_y - increase_bounding_box_per * h), 0)
            w_big = int(w + increase_bounding_box_per * w * 2)
            h_big = int(h + increase_bounding_box_per * h * 2)

            sub_face = image[y_big:y_big + h_big, x_big:x_big + w_big]
            sub_face = cv2.resize(sub_face, (face_config.FACE_FOR_LANDMARKS_IMAGE_RESIZE, face_config.FACE_FOR_LANDMARKS_IMAGE_RESIZE))

            tl_xy = (top_left_x, top_left_y)
            br_xy = (top_left_x + w, top_left_y + h)

            results_sub_face = face_mesh.process(sub_face)
            if results_sub_face.multi_face_landmarks is None:
                if print_flag:
                    print('No mesh detected')
                return None

            for face_landmarks_ in results_sub_face.multi_face_landmarks:
                # print('face_landmarks:', face_landmarks)

                sub_face_landmark_list = []
                # Append all landmarks to list
                for i in range(468):
                    sub_face_landmark_list.append(face_landmarks_.landmark[i].x)
                    sub_face_landmark_list.append(face_landmarks_.landmark[i].y)
                    sub_face_landmark_list.append(face_landmarks_.landmark[i].z)

                disp_img = sub_face.copy()
                if return_mesh:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())


        if print_flag and showImg:
            print('landmark_list:', sub_face_landmark_list[:10])
            print('distances len:', len(sub_face_landmark_list))
            print('landmark_list len:', len(landmark_list))
            print('depth_list:', len(depth_list))

        if showImg:
            Image.fromarray(sub_face_mini).show()

        if save_path is not None:
            cv2.imwrite(save_path, image)

    if return_mesh:
        return sub_face_landmark_list, sub_face_mini, image, (tl_xy, br_xy)

    return sub_face_landmark_list, sub_face_mini

if __name__ == '__main__':
    image = cv2.imread(config.INPUT_FOLDER_PATH + 'grp_smile.jpg')
    get_mesh(image, showImg=True, upscale_landmarks=False, print_flag=True)