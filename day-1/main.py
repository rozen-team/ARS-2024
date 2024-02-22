import copy
import math
import time as t
from enum import Enum
from typing import Any, List, Set, Tuple, Union

USING_IMUTILS = True
try:
    import imutils
except:
    USING_PYMURAPI = False

    print("Imutils disabled.")

import cv2
import numpy as np

USING_PYMURAPI = True
try:
    import pymurapi as mur

    print("Using PyMURAPI")
except:
    USING_PYMURAPI = False
    print("PyMURAPI disabled.")


class Convert:
    @staticmethod
    def list_elements_to(list: List, type: Any) -> List[Any]:
        return [type(i) for i in list]


class Maath:
    @staticmethod
    def vectors2_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return abs(math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2))

    @staticmethod
    def vectors2_triangle_center(
        a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
    ) -> Tuple[float, float]:
        return (
            (a[0] + b[0] + c[0]) / 3,
            (a[1] + b[1] + c[1]) / 3,
        )

    @staticmethod
    def vectors2_square_center(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
        d: Tuple[float, float],
    ) -> Tuple[float, float]:
        return (
            int((a[0] + b[0] + c[0] + d[0]) / 4),
            int((a[1] + b[1] + c[1] + d[1]) / 4),
        )

    @staticmethod
    def vectors2_two_center(
        a: Tuple[float, float], b: Tuple[float, float]
    ) -> Tuple[float, float]:
        return (
            int((a[0] + b[0]) / 2),
            int((a[1] + b[1]) / 2),
        )

    @staticmethod
    def vectors2_angle_sin(a, b):
        # вычесляем угол
        x2 = 0 - 0
        y2 = 0 - 240
        try:
            angle = int(
                math.asin(
                    (a * y2 - b * x2)
                    / (((a**2 + b**2) ** 0.5) * ((x2**2 + y2**2) ** 0.5))
                )
                * 180
                / math.pi
            )
        except ZeroDivisionError:
            angle = int(
                math.asin(
                    (a * y2 - b * x2)
                    / (((a**2 + b**2) ** 0.5 + 0.000001) * ((x2**2 + y2**2) ** 0.5))
                )
                * 180
                / math.pi
            )
        return angle

    @staticmethod
    def vectors2_angle_cos(a, b):
        x2 = 0 - 0
        y2 = 0 - 240
        # вычесляем угол
        angle = int(
            math.acos(
                (a * x2 + b * y2) / (((a**2 + b**2) ** 0.5) * ((x2**2 + y2**2) ** 0.5))
            )
            * 180
            / math.pi
        )
        return angle

    @staticmethod
    def vectors2_angle(a, b):
        angle_ = Maath.vectors2_angle_cos(a, b)
        angle = Maath.vectors2_angle_sin(a, b)
        try:
            angle = angle_ * (angle / abs(angle))
        except ZeroDivisionError:
            pass
        return angle

    @staticmethod
    def vector_abs(a):
        return math.sqrt(a[0] ** 2 + a[1] ** 2)

    @staticmethod
    def dot_product(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def cos_between_vectors(a, b):
        return Maath.dot_product(a, b) / (Maath.vector_abs(a) * Maath.vector_abs(b))


class ShapeType(Enum):
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2
    UNKNOWN = 3
    SQUARE = 4
    FISH = 5
    TREE = 6


ALL_SHAPES = [
    ShapeType.CIRCLE,
    ShapeType.RECTANGLE,
    ShapeType.TRIANGLE,
    ShapeType.UNKNOWN,
    ShapeType.SQUARE,
    ShapeType.TREE,
    ShapeType.FISH,
]


class Color:
    def __init__(self, x: float, y: float, z: float, name: str = ...) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def to_tuple(self) -> Tuple:
        return (self.x, self.y, self.z)


class CompareType(Enum):
    EQUALS = 0
    GREATER = 1
    GREATER_EQUALS = 2
    LOWER = 3
    LOWER_EQUALS = 4
    NOT_EQUALS = 5


class PropertyCompare:
    def __init__(self, name: str, value: Any, compare_type: CompareType) -> None:
        self.name = name
        self.value = value
        self.compare_type = compare_type


class Property:
    def __init__(self, name: str, type: Any = ...) -> None:
        self.name = name

    def __eq__(self, o: Any) -> PropertyCompare:  # ==
        return PropertyCompare(self.name, o, CompareType.EQUALS)

    def __gt__(self, o: Any) -> PropertyCompare:  # >
        return PropertyCompare(self.name, o, CompareType.GREATER)

    def __ge__(self, o: Any) -> PropertyCompare:  # >=
        return PropertyCompare(self.name, o, CompareType.GREATER_EQUALS)

    def __lt__(self, o: Any) -> PropertyCompare:  # <
        return PropertyCompare(self.name, o, CompareType.LOWER)

    def __le__(self, o: Any) -> PropertyCompare:  # <=
        return PropertyCompare(self.name, o, CompareType.LOWER_EQUALS)

    def __ne__(self, o: Any) -> PropertyCompare:  # !=
        return PropertyCompare(self.name, o, CompareType.NOT_EQUALS)

    # def __or__(self, o: 'Property') -> PropertyCompare:


class AnchorType(Enum):
    TOP = 1
    CENTER = 2


class Digit:
    value = Property("shape")
    global_rect = Property("global_rect")
    local_rect = Property("local_rect")

    def __init__(
        self,
        value: int = None,
        global_rect: List[int] = None,
        local_rect: List[int] = None,
    ) -> None:
        self.value = value
        self.global_rect = global_rect
        self.local_rect = local_rect

    def full_rect(self) -> List[int]:
        rect = self.global_rect if self.global_rect is not None else [0] * 4
        return [
            self.local_rect[0] + rect[0],
            self.local_rect[1] + rect[1],
            self.local_rect[2] + rect[0],
            self.local_rect[3] + rect[1],
        ]

    def draw_bounding_box(
        self,
        draw: np.ndarray,
        color: Color,
        thickness: int = 1,
    ):
        rect = self.full_rect()
        cv2.rectangle(
            draw,
            (rect[0], rect[1]),
            (rect[2], rect[3]),
            color.to_tuple() if type(color) == Color else color,
            thickness=thickness,
        )

    def put_text(
        self,
        draw: np.ndarray,
        color: Color,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale: int = 1,
        thickness: int = 2,
        paddings: Tuple[int, int] = [0, 0],
    ):
        rect = self.full_rect()
        cv2.putText(
            draw,
            str(self.value),
            (rect[0] - paddings[0], rect[1] - paddings[1]),
            font,
            scale,
            color.to_tuple() if type(color) == Color else color,
            thickness=thickness,
        )


class Figure:
    shape = Property("shape")
    color_range = Property("color_range")
    color_name = Property("color_name")
    center = Property("center")
    center_x = Property("center_x")
    center_y = Property("center_y")
    rect_area = Property("rect_area")
    circle_area = Property("circle_area")
    triangle_area = Property("triangle_area")
    moments = Property("moments")
    rect_box = Property("rect_box")
    triangle_box = Property("triangle_box")
    circle_radius = Property("circle_radius")
    cnt = Property("cnt")
    cnt_area = Property("cnt_area")
    arrow_angle = Property("arrow_angle")
    vertex = Property("vertex")
    is_convex = Property("is_convex")
    hull = Property("hull")
    hull_defects = Property("hull_defects")

    _allowed_shapes = []

    def __init__(
        self,
        shape: ShapeType = ShapeType.UNKNOWN,
        color_range: "ColorRange" = None,
        color_name: str = None,
        center: Tuple[int, int] = None,
        cnt: np.ndarray = None,
        cnt_area: float = None,
        allowed_shapes: List[ShapeType] = ALL_SHAPES,
        convexity_defects_min_distance: int = 5000,
    ) -> None:
        self._allowed_shapes = allowed_shapes
        self._convexity_defects_min_distance = convexity_defects_min_distance
        self.shape = shape
        self.color_range = color_range
        self.color_name = color_name
        self.center = center

        self.center_x = None
        self.center_y = None
        if self.center is not None:
            self.center_x = center[0]
            self.center_y = center[1]

        self.rect_area = None
        self.circle_area = None
        self.triangle_area = None
        self.moments = None

        self.rect_box = None
        self.triangle_box = None
        self.circle_radius = None

        self.cnt = cnt
        self.cnt_area = cnt_area

        self.arrow_angle = None

    def define_moments(self, cnt: np.ndarray = ...):
        if cnt is not ...:
            self.cnt = cnt
        self.moments = cv2.moments(self.cnt)

    def define_center(self, cnt: np.ndarray = ...) -> bool:
        if cnt is not ...:
            self.cnt = cnt
            self.define_moments()
        try:
            x = int(self.moments["m10"] / self.moments["m00"])
            y = int(self.moments["m01"] / self.moments["m00"])
            self.center = (x, y)
            return True
        except ZeroDivisionError:
            return False

    def allowed(self, shape: ShapeType):
        return shape in self._allowed_shapes

    def define_shape(self, cnt: np.ndarray = ...):
        if cnt is not ...:
            self.cnt = cnt

        (cX, xY), cR = cv2.minEnclosingCircle(self.cnt)
        s_triangle, triangle = cv2.minEnclosingTriangle(self.cnt)
        rect = cv2.minAreaRect(self.cnt)
        ((_, _), (w, h), _) = rect

        self.vertex = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        self.hull = cv2.convexHull(cnt, returnPoints=False)

        try:  # for reason of cv2.error: OpenCV(4.8.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\convhull.cpp:360: error: (-5:Bad argument) The convex hull indices are not monotonous, which can be in the case when the input contour contains self-intersections in function 'cv::convexityDefects'
            self.hull_defects = cv2.convexityDefects(cnt, self.hull)
        except:
            # print(
            #     "WARNING! The convex hull indices are not monotonous. Skipping this contour."
            # )
            self.hull_defects = np.array([])

        self.hull_defects = (
            np.array(
                [
                    self.hull_defects[i, 0]
                    for i in range(self.hull_defects.shape[0])
                    if self.hull_defects[i, 0][3] > self._convexity_defects_min_distance
                ]
            )
            if self.hull_defects is not None
            else np.array([])
        )
        self.is_convex = len(self.hull_defects) == 0
        # self.hull_defects = np.where(self.hull_defects[:, 0][3])

        rectangle_area = w * h
        circle_area = cR**2 * math.pi

        if self.is_convex:
            if (
                self.allowed(ShapeType.CIRCLE)
                and rectangle_area > circle_area * 0.8 < s_triangle
            ):
                self.shape = ShapeType.CIRCLE
                self.circle_radius = cR
            elif circle_area > rectangle_area < s_triangle:
                if self.allowed(ShapeType.SQUARE) and 0.5 < w / h < 1.5:
                    self.shape = ShapeType.SQUARE
                elif self.allowed(ShapeType.RECTANGLE):
                    self.shape = ShapeType.RECTANGLE
                self.rect_box = rect
            elif (
                self.allowed(ShapeType.TRIANGLE)
                and rectangle_area > s_triangle < circle_area
            ):
                self.shape = ShapeType.TRIANGLE
                self.triangle_area = s_triangle
                self.triangle_box = triangle
        elif len(self.hull_defects) == 2:
            s, e, f, d = max(self.hull_defects, key=lambda x: x[3])
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            vec_fs = (far[0] - start[0], far[1] - start[1])
            vec_fe = (far[0] - end[0], far[1] - end[1])

            angle = math.degrees(math.acos(Maath.cos_between_vectors(vec_fs, vec_fe)))
            if self.allowed(ShapeType.TREE) and angle < 90:
                self.shape = ShapeType.TREE
            elif self.allowed(ShapeType.FISH):
                self.shape = ShapeType.FISH

        self.rect_area = rectangle_area
        self.triangle_area = s_triangle
        self.circle_area = circle_area

        self.center_x = self.center[0]
        self.center_y = self.center[1]

        return True

    def draw_contour(
        self, rgb: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0), thickness=1
    ) -> np.ndarray:
        cv2.drawContours(rgb, [self.cnt], 0, color, thickness=thickness)

    def draw_rect(
        self, rgb: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0), thickness=1
    ) -> np.ndarray:
        box = cv2.boxPoints(self.rect_box)
        box = np.int0(box)
        cv2.drawContours(rgb, [box], 0, color, thickness)

    def draw_circle(
        self, rgb: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0), thickness=1
    ) -> np.ndarray:
        cv2.circle(
            rgb,
            (int(self.center[0]), int(self.center[1])),
            int(self.circle_radius),
            color,
            thickness,
        )

    def draw_triangle(
        self, rgb: np.ndarray, color: Tuple[int, int, int] = (0, 0, 0), thickness=1
    ) -> np.ndarray:
        cv2.line(
            rgb,
            tuple(self.triangle_box[0][0]),
            tuple(self.triangle_box[1][0]),
            color,
            thickness,
        )
        cv2.line(
            rgb,
            tuple(self.triangle_box[1][0]),
            tuple(self.triangle_box[2][0]),
            color,
            thickness,
        )
        cv2.line(
            rgb,
            tuple(self.triangle_box[0][0]),
            tuple(self.triangle_box[2][0]),
            color,
            thickness,
        )

    def draw_figure(self, rgb: np.ndarray, color: Color, thickness=1) -> np.ndarray:
        if self.shape == ShapeType.CIRCLE:
            self.draw_circle(
                rgb, color.to_tuple() if type(color) == Color else color, thickness
            )
        elif self.shape == ShapeType.RECTANGLE or self.shape == ShapeType.SQUARE:
            self.draw_rect(
                rgb, color.to_tuple() if type(color) == Color else color, thickness
            )
        elif self.shape == ShapeType.TRIANGLE:
            self.draw_triangle(
                rgb, color.to_tuple() if type(color) == Color else color, thickness
            )
        else:
            self.draw_contour(
                rgb, color.to_tuple() if type(color) == Color else color, thickness
            )

    def define_arrow_angle(self):
        angle_arrow = self.triangle_box
        W1 = (
            ((angle_arrow[1, 0, 0] - angle_arrow[2, 0, 0]) ** 2)
            + ((angle_arrow[1, 0, 1] - angle_arrow[2, 0, 1]) ** 2)
        ) ** 0.5

        W2 = (
            ((angle_arrow[0, 0, 0] - angle_arrow[2, 0, 0]) ** 2)
            + ((angle_arrow[0, 0, 1] - angle_arrow[2, 0, 1]) ** 2)
        ) ** 0.5

        W3 = (
            ((angle_arrow[1, 0, 0] - angle_arrow[0, 0, 0]) ** 2)
            + ((angle_arrow[1, 0, 1] - angle_arrow[0, 0, 1]) ** 2)
        ) ** 0.5

        if W3 < W1 > W2:
            coordinate = [
                [angle_arrow[0, 0, 0], angle_arrow[0, 0, 1]],
                [angle_arrow[1, 0, 0], angle_arrow[1, 0, 1]],
                [angle_arrow[2, 0, 0], angle_arrow[2, 0, 1]],
            ]
        elif W3 < W2 > W1:
            coordinate = [
                [angle_arrow[1, 0, 0], angle_arrow[1, 0, 1]],
                [angle_arrow[2, 0, 0], angle_arrow[2, 0, 1]],
                [angle_arrow[0, 0, 0], angle_arrow[0, 0, 1]],
            ]
        elif W1 < W3 > W2:
            coordinate = [
                [angle_arrow[2, 0, 0], angle_arrow[2, 0, 1]],
                [angle_arrow[0, 0, 0], angle_arrow[0, 0, 1]],
                [angle_arrow[1, 0, 0], angle_arrow[1, 0, 1]],
            ]
        else:
            return False
        # вычисления кооринат середины наибольшой стороны треугольника
        x_centre_arrow = (coordinate[1][0] + coordinate[2][0]) // 2
        y_centre_arrow = (coordinate[1][1] + coordinate[2][1]) // 2
        # вычисление векторов для расчёта угла стрелки
        x1 = coordinate[0][0] - x_centre_arrow
        y1 = coordinate[0][1] - y_centre_arrow
        # вычисляем угол
        angle = Maath.vectors2_angle(x1, y1)
        self.arrow_angle = angle
        return True

    def put_text(
        self,
        rgb: np.ndarray,
        text: str,
        color: Color,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale: int = 1,
        thickness: int = 2,
        anchor: AnchorType = AnchorType.CENTER,
        paddings: Tuple[int, int] = [0, 0],
    ):
        if anchor == AnchorType.TOP:
            x, y, w, h = cv2.boundingRect(self.cnt)
            point = (x, y)
        else:
            self.define_center()
            point = self.center
        point = (point[0] + paddings[0], point[1] + paddings[1])

        cv2.putText(
            rgb,
            text,
            point,
            font,
            scale,
            color.to_tuple() if type(color) == Color else color,
            thickness=thickness,
        )

    def draw_bounding_box(
        self,
        draw: np.ndarray,
        color: Color,
        thickness: int = 1,
        paddings: Tuple[int, int, int, int] = [0, 0, 0, 0],
    ):
        x, y, w, h = cv2.boundingRect(self.cnt)

        cv2.rectangle(
            draw,
            (x + paddings[0], y + paddings[1]),
            (x + w - paddings[2], y + h - paddings[3]),
            color.to_tuple() if type(color) == Color else color,
            thickness=thickness,
        )


class ColorRange:
    def __init__(
        self,
        min_color: Color = ...,
        max_color: Color = ...,
        name: str = ...,
        subranges: List["ColorRange"] = [],
    ) -> None:
        self.name = name
        self.min_color = min_color
        self.max_color = max_color

        self.min_color.name = name
        self.max_color.name = name

        self.subranges = subranges

    def __add__(self, cr: "ColorRange"):
        return ColorRange(
            self.min_color, self.max_color, self.name, self.subranges + [cr]
        )


class HSVTrackbars:
    """Helper for HSV values arrangement."""

    def __init__(self, winname):
        self.winname = winname
        self._created = False

    def create(self):
        nothing = lambda x: None

        cv2.namedWindow(self.winname)
        cv2.createTrackbar("H Lower", self.winname, 0, 179, nothing)
        cv2.createTrackbar("H Higher", self.winname, 179, 179, nothing)
        cv2.createTrackbar("S Lower", self.winname, 0, 255, nothing)
        cv2.createTrackbar("S Higher", self.winname, 255, 255, nothing)
        cv2.createTrackbar("V Lower", self.winname, 0, 255, nothing)
        cv2.createTrackbar("V Higher", self.winname, 255, 255, nothing)
        self._created = True

    @property
    def is_created(self):
        return self._created

    def get(self):
        hL = cv2.getTrackbarPos("H Lower", self.winname)
        hH = cv2.getTrackbarPos("H Higher", self.winname)
        sL = cv2.getTrackbarPos("S Lower", self.winname)
        sH = cv2.getTrackbarPos("S Higher", self.winname)
        vL = cv2.getTrackbarPos("V Lower", self.winname)
        vH = cv2.getTrackbarPos("V Higher", self.winname)

        lowerRegion = np.array([hL, sL, vL], np.uint8)
        upperRegion = np.array([hH, sH, vH], np.uint8)

        return lowerRegion, upperRegion

    def process(self, bgr: np.ndarray):
        if not self.is_created:
            return

        low, high = self.get()
        o = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        o = cv2.morphologyEx(
            o, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        cv2.imshow(
            self.winname,
            cv2.inRange(o, low, high),
        )


class FiguresSearchParams:
    def __init__(
        self,
        color_ranges: List[ColorRange] = ...,
        contours_find_mode: int = cv2.RETR_EXTERNAL,
        contours_find_method: int = cv2.CHAIN_APPROX_SIMPLE,
        min_contour_area: float = 0,
        allowed_shapes: List[ShapeType] = ALL_SHAPES,
        convexity_defects_min_distance=5000,
    ) -> None:
        self.color_ranges = color_ranges
        self.contours_mode = contours_find_mode
        self.min_contour_area = min_contour_area
        self.contours_find_method = contours_find_method
        self.allowed_shapes = allowed_shapes
        self.convexity_defects_min_distance = convexity_defects_min_distance


class DigitsSearchParams:
    def __init__(
        self,
        color_range: ColorRange = ...,
        kernel_size: int = 3,
        one_ratio: int = 3.7,
        segwhite_ratio: float = 0.7,
        height_error: int = 30,
        digit_width: int = 64,
        roi_rect: List[int] = None,
        min_contour_area: float = 150,
    ) -> None:
        self.color_range = color_range
        self.kernel_size = kernel_size
        self.one_ratio = one_ratio
        self.segwhite_ratio = segwhite_ratio
        self.height_error = height_error
        self.roi_rect = roi_rect
        self.digit_width = digit_width
        self.min_contour_area = min_contour_area


class FrameSearchParams:
    def __init__(
        self,
        threshold: int = 10,
        underline_aspect_ratio: float = 8,
        underline_width_coefficient: float = 0.9,
        opening_kernel_size: Tuple[int, int] = (5, 5),
        top_crop: float = 0.2,
        bottom_crop: float = 0,
        min_contour_area: int = 100,
    ):
        self.threshold = threshold
        self.underline_aspect_ratio = underline_aspect_ratio
        self.underline_width_coefficient = underline_width_coefficient
        self.opening_kernel_size = opening_kernel_size
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
        self.min_contour_area = min_contour_area


class FrameSearchResult:
    def __init__(
        self, box: Union[Tuple[int, int, int, int], None], cnt: Union[np.ndarray, None]
    ) -> None:
        self.box = box
        self.cnt = cnt

    @property
    def is_found(self):
        return self.box is not None


class FiguresSearchResult:
    def __init__(
        self, figures: List[Figure] = None, binaries: List["BinarizedImage"] = []
    ) -> None:
        if figures is not None:
            self.raw_figures = figures
        self.figures = FiguresList(self.raw_figures)
        self.binaries = binaries

    def has_any_figure(self) -> bool:
        return len(self.figures) > 0


class DigitsSearchResult:
    def __init__(
        self,
        digits: List[Digit] = None,
    ) -> None:
        self.digits = digits

    def has_any_digit(self) -> bool:
        return len(self.digits) > 0


class FiguresListPointer:
    def __init__(
        self, index: int, figures_list: "FiguresList", got: bool = True
    ) -> None:
        self.index = index
        self.figures_list = figures_list
        self.got = got

    def to_value(self) -> Figure:
        return self.figures_list._figures[self.index]

    def to_value_or_default(self, default: Any = None) -> Union[Figure, Any]:
        if not self.got:
            return default
        return self.to_value()

    @property
    def is_bad_index(self) -> bool:
        return True if self.index < 0 else False


class FunctionResult:
    def __init__(self, function, args=[], kwargs={}, calc: bool = False) -> None:
        self.function = function
        self.args = args
        self.kwargs = kwargs
        if calc:
            self.calculate()
        else:
            self.result = None
            self.calculated = False

    def calculate(self):
        self.result = self.function(*self.args, **self.kwargs)
        self.calculated = True

    def get_args(self) -> List:
        return self.args

    def __str__(self) -> str:
        return "<FunctionResult, calculated={}, result={}, function={}, args={}, kwargs={}>".format(
            self.calculated, self.result, self.function, self.args, self.kwargs
        )

    def __repr__(self) -> str:
        return self.__str__()


# class FunctionResultsList:
#     def __init__(self, function_results: List[FunctionResult]) -> None:
#         self._list = function_results
#     def max(self, func = ...) -> FunctionResult:
#         if func is None:
#             func = lambda x: x.result
#         return


def unpoint(figures_list: "FiguresList"):
    for p in figures_list:
        yield p.to_value()


def extract(figures_list: "FiguresList"):
    for p in figures_list:
        yield (p, p.to_value())


class FiguresList:
    def __init__(
        self,
        figures_list: List[Figure],
        sub_lists: List["FiguresList"] = [],
        function_results: List[FunctionResult] = [],
    ) -> None:
        self._figures = figures_list
        # self._sub_lists = sub_lists
        # self.zipped = None
        # self.function_results = function_results

    def __add__(self, fl: "FiguresList"):
        return FiguresList(self.to_list() + fl.to_list())

    def where(self, *args: PropertyCompare, **kwargs) -> "FiguresList":
        return FiguresList(
            [
                i
                for i in self._figures
                if (
                    all(
                        [
                            (
                                (getattr(i, j.name) == j.value)
                                if j.compare_type == CompareType.EQUALS
                                else (
                                    (getattr(i, j.name) > j.value)
                                    if j.compare_type == CompareType.GREATER
                                    else (
                                        (getattr(i, j.name) >= j.value)
                                        if j.compare_type == CompareType.GREATER_EQUALS
                                        else (
                                            (getattr(i, j.name) < j.value)
                                            if j.compare_type == CompareType.LOWER
                                            else (
                                                (getattr(i, j.name) <= j.value)
                                                if j.compare_type
                                                == CompareType.LOWER_EQUALS
                                                else (
                                                    (getattr(i, j.name) != j.value)
                                                    if j.compare_type
                                                    == CompareType.NOT_EQUALS
                                                    else False
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            for j in args
                        ]
                    )
                )
            ]
        )

    def max(self, key: Property) -> FiguresListPointer:
        if len(self._figures) <= 0:
            return FiguresListPointer(-1, self, False)
        return FiguresListPointer(
            self._figures.index(max(self._figures, key=lambda x: getattr(x, key.name))),
            self,
        )

    def min(self, key: Property) -> FiguresListPointer:
        if len(self._figures) <= 0:
            return FiguresListPointer(-1, self, False)
        return FiguresListPointer(
            self._figures.index(min(self._figures, key=lambda x: getattr(x, key.name))),
            self,
        )

    def sort(self, key: Property, reverse=False) -> "FiguresList":
        return FiguresList(
            list(
                sorted(
                    self._figures, key=lambda x: getattr(x, key.name), reverse=reverse
                )
            )
        )

    def to_list(self) -> List[Figure]:
        return self._figures

    def __len__(self) -> int:
        return len(self._figures)

    def append(self, fig: Figure) -> "FiguresList":
        self._figures.append(fig)
        return self

    def first_or_default(self, *args: PropertyCompare) -> FiguresListPointer:
        for i, el in enumerate(self._figures):
            if all(
                [
                    (
                        (getattr(el, j.name) == j.value)
                        if j.compare_type == CompareType.EQUALS
                        else (
                            (getattr(el, j.name) > j.value)
                            if j.compare_type == CompareType.GREATER
                            else (
                                (getattr(el, j.name) >= j.value)
                                if j.compare_type == CompareType.GREATER_EQUALS
                                else (
                                    (getattr(el, j.name) < j.value)
                                    if j.compare_type == CompareType.LOWER
                                    else (
                                        (getattr(el, j.name) <= j.value)
                                        if j.compare_type == CompareType.LOWER_EQUALS
                                        else (
                                            (getattr(el, j.name) != j.value)
                                            if j.compare_type == CompareType.NOT_EQUALS
                                            else False
                                        )
                                    )
                                )
                            )
                        )
                    )
                    for j in args
                ]
            ):
                return FiguresListPointer(i, self)
        return FiguresListPointer(-1, self, False)

    def pop(self, index: int) -> "FiguresList":
        self._figures.pop(index)
        return self

    def remove(self, fig: Figure) -> "FiguresList":
        self._figures.remove(fig)
        return self

    def __getitem__(self, index: int) -> FiguresListPointer:
        size = self.length
        return FiguresListPointer(index, self, index < size)

    # def __setitem__(self, index: int, fig: Figure) -> None:
    #     self._figures[index] = fig

    def to_tuple(self) -> Tuple[Figure]:
        return tuple(self._figures)

    def __iter__(self):
        for i, el in enumerate(self._figures):
            yield FiguresListPointer(i, self)

    @property
    def length(self):
        return self.__len__()

    # def with_other(self, other_list: 'FiguresList') -> 'FiguresList':
    #     self._sub_lists.append(other_list)
    #     return self

    # def zip(self, dims: Set[int] = ...):
    #     if dims is not ...:
    #         self.zipped = dims
    #     else:
    #         self.zipped = {i for i in range(len(self._sub_lists) + 1)}
    #     return self

    # def _zzip(self):
    #     zziped = []
    #     for el in self._figures:
    #         zziped.append([el])
    #     for i, el in enumerate(self._sub_lists):
    #         if i > 0:
    #             for el2 in unpoint(el):
    #                 zziped[i].append(el2)
    #         # else:
    #         #     for el2 in unpoint(self):
    #         #         zziped[i].append(el2)
    #     return zziped

    # def exec_func(self, func):
    #     z = self._zzip()
    #     for elements in z:
    #         self.function_results.append(FunctionResult(func, elements))
    #     return self

    # def calc_results(self):
    #     for f in self.function_results:
    #         f.calculate()
    #     return [f.result for f in self.function_results]

    def copy(self) -> "FiguresList":
        return copy.deepcopy(self)

    def compose(self, lists: List["FiguresList"]) -> "FiguresListComposition":
        allllllll = [self, *lists]
        return FiguresListComposition(allllllll)

    # def self_unique_compose(self) -> 'FiguresListUniqueComposition':
    #     ...


class FiguresListComposition:
    def __init__(self, fig_lists: List[FiguresList]) -> None:
        self._lists = fig_lists

    def _recursive_list(self, lists: List[FiguresList]):
        if len(lists) == 1:
            return [[i.to_value()] for i in lists[0]]
        l = []
        for other in self._recursive_list(lists[1:]):
            for i in lists[0]:
                l.append((i.to_value(), *other))
        return l

    def exec_func(
        self, func, calc: bool = True, unpoint: bool = True
    ) -> List[FunctionResult]:
        a = self._recursive_list(self._lists)
        # new = []
        # for elements in zip(*self._lists):
        #     if unpoint:
        #         elements = [i.to_value() for i in elements]
        #     new.append(FunctionResult(func, elements, calc=calc))
        # return new
        return [FunctionResult(func, i, calc=calc) for i in a]


# class FiguresListUniqueComposition:
#     def _recursive_list(self, lists: List[FiguresList]):
#         if len(lists) == 1:
#             return [[i.to_value()] for i in lists[0]]
#         l = []
#         for other in self._recursive_list(lists[1:]):
#             for i in lists[0]:
#                 l.append((i.to_value(), *other))
#         return l


class BinarizedImage:
    def __init__(self, img: np.ndarray, color_range: ColorRange) -> None:
        self.img = img
        self.color_range = color_range


def find_figures(rgb: np.ndarray, params: FiguresSearchParams) -> FiguresSearchResult:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    figures = []
    bimages = []
    for range in params.color_ranges:
        color_min = range.min_color.to_tuple()
        color_max = range.max_color.to_tuple()
        bin = cv2.inRange(hsv, color_min, color_max)

        for sr in range.subranges:
            color_min = sr.min_color.to_tuple()
            color_max = sr.max_color.to_tuple()
            bin |= cv2.inRange(hsv, color_min, color_max)

        bimages.append(BinarizedImage(bin, range))

        contours, _ = cv2.findContours(
            bin, params.contours_mode, params.contours_find_method
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > params.min_contour_area:
                fig = Figure(
                    color_range=range,
                    cnt=cnt,
                    cnt_area=area,
                    color_name=range.name,
                    allowed_shapes=params.allowed_shapes,
                    convexity_defects_min_distance=params.convexity_defects_min_distance,
                )
                if not fig.define_center(cnt):
                    continue  # if no center found. HOW? - yes!
                fig.define_shape(cnt)
                figures.append(fig)
    return FiguresSearchResult(figures, bimages)


def classify_digit(
    bin: np.ndarray, rect: List[int], params: DigitsSearchParams
) -> Digit:
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
    }

    roi = bin[rect[1] : rect[3], rect[0] : rect[2]].copy()
    (h, w) = roi.shape

    # detect the digit --> 1
    # move out this value to params
    if (h / w) > params.one_ratio:
        return Digit(value=1, global_rect=params.roi_rect, local_rect=rect)

    roi = cv2.resize(roi, (params.digit_width, int(params.digit_width / 0.6)))
    (h, w) = roi.shape

    (dW, dH) = (int(w * 0.25), int(h * 0.15))
    dHC = int(h * 0.05)
    dBrd = int(w * 0.1)
    segments = [
        ((dBrd, dBrd), (w - dBrd, dH + dBrd)),  # top
        ((dBrd, dBrd), (dW + dBrd, h // 2)),  # top-left
        ((w - dW - dBrd, dBrd), (w - dBrd, h // 2)),  # top-right
        ((dBrd, (h // 2) - dHC), (w - dBrd, (h // 2) + dHC)),  # center
        ((dBrd, h // 2), (dW + dBrd, h - dBrd)),  # bottom-left
        ((w - dW - dBrd, h // 2), (w - dBrd, h - dBrd)),  # bottom-right
        ((dBrd, h - dH - dBrd), (w - dBrd, h - dBrd)),  # bottom
    ]
    on = [0] * len(segments)

    for i, ((xA, yA), (xB, yB)) in enumerate(segments):
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        cv2.rectangle(roi, (xA, yA), (xB, yB), 127, 1)
        area = (xB - xA) * (yB - yA)
        if total / (float(area) + pow(10, -10)) > params.segwhite_ratio:
            on[i] = 1

    if tuple(on) not in DIGITS_LOOKUP:
        return None
    return Digit(
        value=DIGITS_LOOKUP[tuple(on)], global_rect=params.roi_rect, local_rect=rect
    )


def find_digits(rgb: np.ndarray, params: DigitsSearchParams) -> DigitsSearchResult:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

    roi = hsv.copy()
    if params.roi_rect is not None:
        # crop roi
        roi_rect = params.roi_rect
        roi = hsv[roi_rect[1] : roi_rect[3], roi_rect[0] : roi_rect[2]]

    rng = params.color_range
    min_color = rng.min_color.to_tuple()
    max_color = rng.max_color.to_tuple()
    bin = cv2.inRange(roi, min_color, max_color)

    # a bit of filters...
    ksize = params.kernel_size
    kernel = np.ones((ksize, ksize), np.uint8)
    bin = cv2.erode(bin, kernel)

    # hardcode the parameters of findContoures. We always search external cnt
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    digits = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= params.min_contour_area:
            continue

        # if in one cnt consist of several digits: split them manually
        x, y, w, h = cv2.boundingRect(cnt)
        digits_count = round((w / h) * 2)
        step = w / digits_count
        for i in range(digits_count):
            digit_rect = [x + i * step, y, x + min((i + 1) * step, w), y + h]
            digit_rect = list(map(int, digit_rect))
            digit_value = classify_digit(bin, digit_rect, params)

            if digit_value is not None:
                digits.append(digit_value)

        if digits_count == 0:
            # check full roi (when one digit in it)
            digit_rect = [x, y, x + w, y + h]
            digit_value = classify_digit(bin, digit_rect, params)
            if digit_value is not None:
                digits.append(digit_value)

    return DigitsSearchResult(digits)


def find_frame(rgb: np.ndarray, params: FrameSearchParams) -> FrameSearchResult:
    """Finds frame on image based on black underline, adds underline width, multiplied by a coefficient, as -y to the underline box.

    Args:
        rgb (np.ndarray): BGR image.
        params (FrameSearchParams): Search parameters.

    Returns:
        FrameSearchResult: search result.
    """
    _, thresh = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        params.threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params.opening_kernel_size),
    )
    sl = int(thresh.shape[0] * params.top_crop)
    thresh = thresh[sl:, :]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = []
    if len(cnts) > 0:
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            if (
                cv2.contourArea(c) > params.min_contour_area
                and w / h > params.underline_aspect_ratio
            ):
                cs.append(c)
    if len(cs) > 0:
        c = max(cs, key=lambda x: cv2.boundingRect(x)[1])
        x, y, w, h = cv2.boundingRect(c)
        h += int(w * params.underline_width_coefficient)
        y -= int(w * params.underline_width_coefficient)
        y += sl
        return FrameSearchResult((x, y, w, h), c)
    return FrameSearchResult(None, None)


def focus(bgr: np.ndarray, box: Tuple[int, int, int, int]):
    """Takes a defined region from an original image and pastes this region to the new black image with the same size as original.

    Args:
        bgr (np.ndarray): BGR image.
        box (Tuple[int, int, int, int]): A region to focus.

    Returns:
        np.ndarray: new image.
    """
    x, y, w, h = box
    newimage = np.zeros(bgr.shape, np.uint8)
    newimage[max(y, 0) : y + h, x : x + w, :] = image[max(y, 0) : y + h, x : x + w, :]
    return newimage


class PD:
    _kp = 0.0
    _kd = 0.0
    _prev_error = 0.0
    _timestamp = 0

    def __init__(self, kp: float = ..., kd: float = ...):
        if kp is not ...:
            self.set_p_gain(kp)
        if kd is not ...:
            self.set_d_gain(kd)

    def set_p_gain(self, value):
        self._kp = value

    def set_d_gain(self, value):
        self._kd = value

    def process(self, error):
        timestamp = int(round(t.time() * 1000))  # в timestamp записываем
        try:
            # время(выраженное в секундах) и домножаем до милисекунд, round отбрасывает знаки после запятой
            output = self._kp * error + self._kd / (timestamp - self._timestamp) * (
                error - self._prev_error
            )
            # вычесляем выходное значение на моторы по ПД регулятору и записываем в output
            self._timestamp = timestamp  # перезаписываем время
            self._prev_error = error  # перезаписываем ошибку
            return output
        except ZeroDivisionError:
            return 0


class Functions:
    @staticmethod
    def opencv_show_func(frame, window_name: str = "window"):
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    @staticmethod
    def simulator_get_front_frame_func(auv: "SmartAUV"):
        return auv.get_image_front()

    @staticmethod
    def simulator_get_bottom_frame_func(auv: "SmartAUV"):
        return auv.get_image_bottom()

    @staticmethod
    def clamp(v: float, min_v: float, max_v: float) -> float:
        """Clamp value between two ranges.

        Args:
            v (float): Value
            min_v (float): Min value
            max_v (float): Max value

        Returns:
            float: Clamped value
        """
        if v > max_v:
            return max_v
        if v < min_v:
            return min_v
        return v

    @staticmethod
    def stabilizate_value_by_time(
        timer: float,
        value_to_set: float,
        current_value: float,
        accuracy: float,
        time_to_keep_value: float,
    ) -> Union[float, bool]:
        """Stabilizate value to be in accuracy range for any time.

        Args:
            timer (float): Timer.
            value_to_set (float): Value must to be that.
            current_value (float): Current value
            accuracy (float): Accuracy range.
            time_to_keep_value (float): For this time value will be kept.

        Returns:
            Union[float, bool]: float - timer. Must be reset for this func after. bool - if stabilizated.
        """
        # print(value_to_set - accuracy, value_to_set + accuracy)
        if (
            value_to_set - accuracy <= current_value
            and current_value <= value_to_set + accuracy
        ):
            if t.time() > timer + time_to_keep_value:
                return timer, True
        else:
            timer = t.time()
        return timer, False


class SmartAUV(mur.simulator.Simulator if USING_PYMURAPI else object):
    def __init__(
        self,
        screen_size=(320, 240),
        get_front_frame_func=Functions.simulator_get_front_frame_func,
        get_bottom_frame_func=Functions.simulator_get_bottom_frame_func,
        show_frame_func=Functions.opencv_show_func,
        yaw_regulator=PD(1, 0.001),
        depth_regulator=PD(10, 5),
        prepare: bool = True,
    ):
        super().__init__()
        self.screen_size = screen_size
        # self.get_frame_func = lambda x: np.zeros((screen_size[1], screen_size[0], 3), np.uint8)
        self.set_get_front_frame_func(get_front_frame_func)
        self.set_get_bottom_frame_func(get_bottom_frame_func)
        self.set_show_frame_func(show_frame_func)
        self.yaw_pd = yaw_regulator
        self.depth_pd = depth_regulator
        self.THRUSTER_YAW_LEFT = 0
        self.THRUSTER_YAW_RIGHT = 1
        self.THRUSTER_DEPTH_LEFT = 2
        self.THRUSTER_DEPTH_RIGHT = 3

        self.THRUSTER_YAW_LEFT_DIRECTION = +1
        self.THRUSTER_YAW_RIGHT_DIRECTION = +1
        self.THRUSTER_DEPTH_LEFT_DIRECTION = +1
        self.THRUSTER_DEPTH_RIGHT_DIRECTION = +1

        if prepare:
            self.prepare()

    # @property
    # def yaw(self):
    #     return self.get_yaw()
    # @property
    # def depth(self):
    #     return self.get_depth()
    def set_thrusters(
        self,
        thruster_yaw_left_num: int = ...,
        thruster_yaw_right_num: int = ...,
        thruster_depth_left_num: int = ...,
        thruster_depth_right_num: int = ...,
        thruster_yaw_left_direction: int = ...,
        thruster_yaw_right_direction: int = ...,
        thruster_depth_left_direction: int = ...,
        thruster_depth_right_direction: int = ...,
    ):
        if thruster_yaw_left_num is not ...:
            self.THRUSTER_YAW_LEFT = thruster_yaw_left_num
        if thruster_yaw_right_num is not ...:
            self.THRUSTER_YAW_RIGHT = thruster_yaw_right_num
        if thruster_depth_left_num is not ...:
            self.THRUSTER_DEPTH_LEFT = thruster_depth_left_num
        if thruster_depth_right_num is not ...:
            self.THRUSTER_DEPTH_RIGHT = thruster_depth_right_num

        if thruster_yaw_left_direction is not ...:
            self.THRUSTER_YAW_LEFT_DIRECTION = thruster_yaw_left_direction
        if thruster_yaw_right_direction is not ...:
            self.THRUSTER_YAW_RIGHT_DIRECTION = thruster_yaw_right_direction
        if thruster_depth_left_direction is not ...:
            self.THRUSTER_DEPTH_LEFT_DIRECTION = thruster_depth_left_direction
        if thruster_depth_right_direction is not ...:
            self.THRUSTER_DEPTH_RIGHT_DIRECTION = thruster_depth_right_direction

    def set_get_front_frame_func(self, func):
        self.get_front_frame_func = func

    def set_get_bottom_frame_func(self, func):
        self.get_bottom_frame_func = func

    def set_show_frame_func(self, func):
        self.show_frame_func = func

    def __enter__(self):
        return self.get_front_frame_func(self), self.get_bottom_frame_func(self)

    def __exit__(self, *args, **kwargs): ...

    def show(self, frame):
        self.show_frame_func(frame)

    def stab_by_arrow(
        self,
        arrow_color_range: ColorRange,
        time_to_stab: float = 5,
        accuracy: float = 5,
        show_video: bool = True,
        draw_arrows: bool = True,
        depth_to_keep: float = 2,
    ) -> float:
        timer = t.time()
        while True:
            # frame = auv.get_image_bottom()
            frame = self.get_front_frame_func(self)
            # print(frame.shape)
            draw = frame.copy()
            f = find_figures(frame, FiguresSearchParams([arrow_color_range])).figures
            triangle_p = f.where(
                Figure.color_name == "orange", Figure.shape == ShapeType.TRIANGLE
            ).max(Figure.cnt_area)
            if not triangle_p.is_bad_index:
                f_triangle = triangle_p.to_value()
                if draw_arrows:
                    f_triangle.draw_figure(draw, thickness=3)
                if f_triangle.define_arrow_angle():
                    print(f_triangle.arrow_angle)
                    timer, stabilizated = Functions.stabilizate_value_by_time(
                        timer,
                        self.get_yaw() + f_triangle.arrow_angle,
                        self.get_yaw(),
                        accuracy,
                        time_to_stab,
                    )
                    if stabilizated:
                        # print("Stabbed!")
                        return
                        # keep(auv.get_yaw() - f_triangle.arrow_angle, 3, time)
                    else:
                        self.keep(
                            self.get_yaw() - f_triangle.arrow_angle,
                            depth_to_keep,
                            time=0.01,
                        )
            if show_video:
                self.show_frame_func(draw)

    def keep_yaw(self, yaw_to_set: float, move_speed: float = 0):
        """Keep robot yaw and move forward/backward.

        Args:
            yaw_to_set (float): Yaw to set
            move_speed (float, optional): Speed to move forward. Defaults to 0.
        """
        err = Functions.clamp(
            self.yaw_pd.process(yaw_to_set - self.get_yaw()), -100, 100
        )
        self.set_motor_power(
            self.THRUSTER_YAW_LEFT, move_speed + err * self.THRUSTER_YAW_LEFT_DIRECTION
        )
        self.set_motor_power(
            self.THRUSTER_YAW_RIGHT,
            move_speed - err * self.THRUSTER_YAW_RIGHT_DIRECTION,
        )
        t.sleep(0.1)

    def keep_depth(self, depth_to_det: float):
        """Keep robot depth.

        Args:
            depth_to_det (float): Depth to set.
        """
        err = Functions.clamp(
            self.depth_pd.process(depth_to_det - self.get_depth()), -100, 100
        )
        self.set_motor_power(
            self.THRUSTER_DEPTH_LEFT, -err * self.THRUSTER_DEPTH_LEFT_DIRECTION
        )
        self.set_motor_power(
            self.THRUSTER_DEPTH_RIGHT, -err * self.THRUSTER_DEPTH_RIGHT_DIRECTION
        )
        t.sleep(0.1)

    def keep(
        self,
        yaw_to_set: float = ...,
        depth_to_set: float = ...,
        move_speed: float = 0,
        time: float = 0,
    ):
        """Keep depth, yaw and move robot forward/backward.

        Args:
            yaw_to_set (float, optional): Yaw to set. Defaults to ....
            depth_to_set (float, optional): Depth to set. Defaults to ....
            move_speed (float, optional): Move forward speed. Defaults to 0.
            time (float, optional): Time to keep. Defaults to 0.
        """
        timer = t.time() + time
        while t.time() < timer:
            if yaw_to_set is not ...:
                self.keep_yaw(yaw_to_set, move_speed)
            else:
                self.set_motor_power(
                    self.THRUSTER_YAW_LEFT,
                    move_speed * self.THRUSTER_YAW_LEFT_DIRECTION,
                )
                self.set_motor_power(
                    self.THRUSTER_YAW_RIGHT,
                    -move_speed * self.THRUSTER_YAW_RIGHT_DIRECTION,
                )
            if depth_to_set is not ...:
                self.keep_depth(depth_to_set)
            t.sleep(0.01)

    def move_to_color(
        self,
        color_range: ColorRange,
        delay_time: int = 0,
        depth_to_keep: float = 3,
        yaw_to_keep: float = ...,
        forward_speed: float = 50,
        show_frame: bool = True,
    ):
        timer = t.time()
        if yaw_to_keep is ...:
            yaw_to_keep = self.get_yaw()
        while True:
            frame = self.get_front_frame_func(self)
            figures = find_figures(frame, FiguresSearchParams([color_range])).figures
            pointer = figures.where(
                Figure.color_name == "orange", Figure.shape == ShapeType.TRIANGLE
            ).max(Figure.cnt_area)
            if not pointer.is_bad_index:
                pointer.to_value().draw_figure(frame)
            timer, stabilizated = Functions.stabilizate_value_by_time(
                timer,
                1,
                0 if pointer.is_bad_index else 1,
                accuracy=0,
                time_to_keep_value=delay_time,
            )
            if stabilizated:
                return
            else:
                self.keep(
                    yaw_to_keep, depth_to_keep, time=0.01, move_speed=forward_speed
                )
            if show_frame:
                self.show_frame_func(frame)

    def forward(
        self,
        time: float,
        yaw_to_keep: float = ...,
        depth_to_keep: float = ...,
        speed: float = 50,
        show_frame: bool = True,
    ):
        timer = t.time() + time
        if yaw_to_keep is ...:
            yaw_to_keep = self.get_yaw()
        if depth_to_keep is ...:
            depth_to_keep = self.get_depth()
        while True:
            if show_frame:
                self.show_frame_func(self.get_front_frame_func(self))
            self.keep(yaw_to_keep, depth_to_keep, speed, 0.01)
            if t.time() > timer:
                return


def batch_test(sequence):
    if not hasattr(batch_test, "index"):
        batch_test.index = 0

    image = cv2.imread(sequence[batch_test.index])

    if USING_IMUTILS:
        image = imutils.resize(image, 320)

    batch_test.index += 1
    if batch_test.index >= len(sequence):
        batch_test.index = 0

    return image


if __name__ == "__main__":
    RANGES_TEST = {
        "blue": ColorRange(Color(90, 100, 0), Color(150, 255, 200), "blue"),
        "red": ColorRange(Color(0, 150, 20), Color(15, 255, 255), "red")
        + ColorRange(Color(170, 150, 20), Color(180, 255, 255), "red"),
        "yellow": ColorRange(Color(15, 0, 0), Color(45, 255, 255), "yellow"),
        "green": ColorRange(Color(45, 0, 0), Color(80, 255, 255), "green"),
    }

    RANGES_ROBOT = {
        "blue": ColorRange(Color(90, 100, 10), Color(150, 255, 200), "blue"),  # ok
        "red": ColorRange(Color(0, 150, 40), Color(15, 255, 255), "red")  # good
        + ColorRange(Color(170, 150, 30), Color(180, 255, 255), "red"),
        "yellow": ColorRange(Color(37, 234, 50), Color(56, 255, 255), "yellow"),  # bad
        "green": ColorRange(Color(57, 204, 43), Color(63, 255, 255), "green"),  # ok
    }

    ranges = RANGES_ROBOT

    whiteColor = Color(255, 255, 255, "white")
    greenColor = Color(0, 255, 0, "green")
    redColor = Color(0, 0, 255, "red")
    blueColor = Color(255, 0, 0, "blue")
    yellowColor = Color(0, 255, 255, "yellow")
    black = Color(0, 0, 0, "black")

    searchParams = FiguresSearchParams(
        [
            ranges["red"],
            ranges["blue"],
            ranges["yellow"],
            ranges["green"],
        ],
        min_contour_area=70,
        allowed_shapes=ALL_SHAPES,
        convexity_defects_min_distance=800,
    )

    digitsSearchParams = DigitsSearchParams(
        color_range=ColorRange(Color(0, 0, 200), Color(180, 100, 255), "white"),
        min_contour_area=70,
        kernel_size=3,
        roi_rect=[107, 20, 250, 170],
    )

    frameSearchParams = FrameSearchParams(threshold=20, underline_width_coefficient=1)

    # sequence = [
    #     "day-1/data/t1.png",
    #     "day-1/data/t2.png",
    #     "day-1/data/t3.png",
    #     "day-1/data/t4.png",
    # ]
    # sequence = [
    #    "day-1/data/tr1.jpg",
    #    "day-1/data/tr2.jpg",
    #    "day-1/data/tr3.jpg",
    #    "day-1/data/tr4.jpg",
    #    "day-1/data/tr5.jpg",
    # ]
    cap = cv2.VideoCapture("day-1/data/vid3-c4.mp4")
    # cap = cv2.VideoCapture("day-1/data/vid2.mp4")

    # test digits
    # sequence = ["day-1/data/digits.png", "day-1/data/digits_1.png"]
    sequence = ["day-1/data/tr7.png"]

    auv = SmartAUV(
        # get_front_frame_func=lambda self: cap.read()[1],
        get_front_frame_func=lambda self: batch_test(sequence),
        get_bottom_frame_func=lambda self: None,
        prepare=False,
    )

    trackbars = HSVTrackbars("HSV Trackbars")
    trackbars.create()

    while True:
        with auv as (image, _):
            draw = image.copy()
            cv2.imshow("img", image)

            digitsSearch = find_digits(image, digitsSearchParams)
            if len(digitsSearch.digits) > 0:
                # find first digit of depth
                start_digit = min(digitsSearch.digits, key=lambda x: x.full_rect()[1])
                # find another digit, where height is similarly to height of start digit
                depth_digits = filter(
                    lambda x: abs(x.full_rect()[1] - start_digit.full_rect()[1])
                    <= digitsSearchParams.height_error,
                    digitsSearch.digits,
                )
                depth_digits = sorted(depth_digits, key=lambda x: x.full_rect()[0])
                # for test
                for digit in depth_digits:
                    digit.draw_bounding_box(draw, redColor, 2)
                    digit.put_text(draw, redColor, scale=0.5, paddings=[0, 5])

                dig_roi = digitsSearchParams.roi_rect
                cv2.rectangle(
                    draw,
                    (dig_roi[0], dig_roi[1]),
                    (dig_roi[2], dig_roi[3]),
                    (0, 50, 255),
                    2,
                )

            frame_result = find_frame(image, frameSearchParams)
            if frame_result.is_found:
                x, y, w, h = frame_result.box
                # print(x, y, w, h)
                cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 255, 255), 2)

                image = focus(image, (x, y, w, h))

                results = find_figures(image, searchParams)
                figures = results.figures.where(Figure.center_y > y + h * 0.1)
                binaries = results.binaries

                for b in binaries:
                    cv2.imshow(b.color_range.name, b.img)

                icefishes = figures.where(Figure.shape == ShapeType.FISH)
                for i in icefishes:
                    i.to_value().put_text(draw, "Fish", black)
                phytoplankton = figures.where(Figure.color_name == "green")
                for i in phytoplankton:
                    i.to_value().put_text(draw, "Plankton", black)

                circles = figures.where(Figure.shape == ShapeType.CIRCLE)
                shellfish_and_sea_urchins = circles.where(
                    Figure.color_name == "red"
                ) + circles.where(Figure.color_name == "yellow")

                for i in shellfish_and_sea_urchins:
                    i.to_value().put_text(draw, "SS", black)

                plants = figures.where(Figure.shape == ShapeType.TREE)
                for i in plants:
                    i.to_value().put_text(draw, "Plant", black)

                rectangles = figures.where(
                    Figure.shape == ShapeType.SQUARE
                ) + figures.where(Figure.shape == ShapeType.RECTANGLE)
                trash = (
                    rectangles.where(Figure.color_name == "blue")
                    + rectangles.where(Figure.color_name == "red")
                    + rectangles.where(Figure.color_name == "yellow")
                    + figures.where(
                        Figure.shape == ShapeType.CIRCLE, Figure.color_name == "blue"
                    )
                )

                all_figures = [
                    ("Icefish", icefishes),
                    ("Phytoplankton", phytoplankton),
                    ("Shellfish and sea urchins", shellfish_and_sea_urchins),
                    ("Plants", plants),
                ]
                all_figures.sort(key=lambda x: len(x[1]), reverse=True)
                all_figures.append(("Trash", trash))
                draw_centers = [i.to_value().center for i in all_figures[0][1]]

                for name, group in all_figures[1:]:
                    for i in group:
                        v = i.to_value()
                        if any([math.dist(c, v.center) < 20 for c in draw_centers]):
                            continue

                        v.draw_bounding_box(
                            draw, redColor, 5, paddings=[-15, -15, -15, -15]
                        )
                        v.put_text(
                            draw,
                            "alien",
                            redColor,
                            anchor=AnchorType.TOP,
                            paddings=[-20, -20],
                        )
                        draw_centers.append(v.center)

                cv2.putText(
                    draw,
                    "Type: " + all_figures[0][0],
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    draw,
                    "Depth: " + "-1",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                print(
                    "Icefish:",
                    len(icefishes),
                    "plankton:",
                    len(phytoplankton),
                    "SS:",
                    len(shellfish_and_sea_urchins),
                    "plants:",
                    len(plants),
                    "trash:",
                    len(trash),
                )

            # cv2.imshow("Thresh", thresh)

            trackbars.process(image)
            auv.show(draw)

            k = cv2.waitKey(100)
            # if k != -1:
            #     print(k)
            if k == 49:
                cv2.imshow("Screenshot", draw)
