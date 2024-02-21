import cv2
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import os


if not os.path.exists("day-2/data/dataset/images"):
    os.makedirs("day-2/data/dataset/images")

tree = ET.parse("day-2/data/annotations.xml")
root = tree.getroot()


@dataclass
class Box:
    coordinates: tuple
    label: str
    index: int


boxes = {}
for img in tree.findall("image"):
    if len(img) > 0:
        i = img[0]
        xtl, ytl, xbr, ybr = [
            round(float(j))
            for j in [i.get("xtl"), i.get("ytl"), i.get("xbr"), i.get("ybr")]
        ]
        label = i[0].text
        boxes[int(img.get("id"))] = Box((xtl, ytl, xbr, ybr), label, int(img.get("id")))

cap = cv2.VideoCapture("day-2/data/vid2.mp4")

indexies = []

i = 0
while True:
    r, frame = cap.read()
    if not r:
        break

    current = boxes.get(i, Box(None, None, -1))
    if i == current.index:
        # print(current)
        xtl, ytl, xbr, ybr = boxes[i].coordinates
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(current.label) != 5:
            print("ACHTGUNG")
            print(i)
            print(current.label)
            break

        # cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (255, 255, 255), 3)

        # cv2.imshow("sample", frame)
        # cv2.waitKey(0)

        indexies.append(f"{i},{current.label},{xtl},{ytl},{xbr},{ybr}")
        cv2.imwrite(f"day-2/data/dataset/images/{i}.jpg", frame[ytl:ybr, xtl:xbr])

    if i % 100 == 0:
        print(f"Proccessed frame {i}")

    i += 1

with open("day-2/data/dataset/labels.txt", "w") as file:
    file.write("\n".join(indexies))
