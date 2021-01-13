import os, dlib, glob

from PIL import Image

crop_size = 150

face_detector = dlib.get_frontal_face_detector()

files = [o for o in glob.iglob(os.path.join("input", '**'), recursive=True) if o.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'))]

print('found %d files' % len(files))

file_count = 1
face_count = 1

for file in files:
    try:
        detected_faces = face_detector(dlib.load_rgb_image(file), 1)
    except:
        continue

    print("[%d of %d] %d detected faces in %s" % (file_count, len(files), len(detected_faces), file))

    for face_rect in detected_faces:
        width = face_rect.right() - face_rect.left()

        height = face_rect.bottom() - face_rect.top()

        if width >= crop_size and height >= crop_size:
            image_to_crop = Image.open(file)
            
            crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())

            cropped_image = image_to_crop.crop(crop_area)

            cropped_image.thumbnail((crop_size, crop_size))

            cropped_image.save(os.path.join("output", str(face_count) + ".jpg"), "JPEG")

            face_count = face_count + 1

    file_count = file_count + 1