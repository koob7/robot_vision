import cv2

if __package__:
    from . import single_determine_pose
    from . import stereo_determine_pose
    from . import vision_helper
else:
    import single_determine_pose
    import stereo_determine_pose
    import vision_helper
import numpy as np
import time

determine_pose = single_determine_pose.single_determine_pose("mx_brio_for_business")

def display_frames(frames):
    if frames is None:
        return

    if isinstance(frames, tuple):
        cv2.imshow(f"{determine_pose.get_name()}_left", frames[0])
        cv2.imshow(f"{determine_pose.get_name()}_right", frames[1])
    else:
        cv2.imshow(determine_pose.get_name(), frames)

    cv2.waitKey(1)


while True:
    found_markers, frames = determine_pose.find_markers()
    display_frames(frames)

    if found_markers:
        pose = vision_helper.get_marker_mutal_pose(found_markers, pairs=[(54, 56)])
        if pose is not None:
            pose_53_50 = pose.get((54, 56))
            if pose_53_50 is not None:
                text = f"R_diff y:{pose_53_50[3]:.2f}° z:{pose_53_50[4]:.2f}°"
                print(text)
                # time.sleep(0.5)
    
    mean_markers, frames = determine_pose.get_mean(10)
    display_frames(frames)
    if mean_markers:
        pose = vision_helper.get_marker_mutal_pose(mean_markers, pairs=[(53, 50)])
        if pose is not None:
            pose_53_50 = pose.get((53, 50))
            if pose_53_50 is not None:
                text = f"Mean R_diff y:{pose_53_50[3]:.2f}° z:{pose_53_50[4]:.2f}°"
                print(text)
                # time.sleep(0.5)

