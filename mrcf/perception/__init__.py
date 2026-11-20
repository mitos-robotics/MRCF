"""
perception — seeing the work surface, from any calibrated camera.

Frames and camera models in, measurements out. Nothing here knows which arm
holds a camera, what the task is, or what the objects are called: the backend
supplies frames and where each camera is (RobotEnv.render_image, camera_model),
a deployment adds what its scene means, and scene_scan.py (top level) turns
measurements into the object memory's and the planner's terms.

  plane_geometry  an outline into a foot, height and width profile, on a plane
  segment         outlines of whatever is there (SAM 2, optional)
  surface_scan    every object on the surface; telling it from the surface itself,
                  in any light (an empty-surface photo, levelled; lined up again
                  when the camera shifts)
  fiducials       AprilTags and bullseyes: where a camera, a gripper, a board is
  hand_eye        a camera's pose on a moving link, from views of a known target
  expected_view   what a located camera should see of objects another camera found:
                  who stands in front of whom, and whose outline is whose
  lighting        is the light good enough: washed out, too dark, area by area, in words
  outlines        one outline per object from a segmenter's many; an outline's middle
  tones           where a colour falls between two of the surface's own, in any light
  tiles           objects cut upright from one frame, numbered on one sheet for a vision
                  model to name all at once

The rules are checked on the source (tests/test_architecture.py): nothing here
imports the world model, the memory, the planner, the arm or a policy.
"""
