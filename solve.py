import numpy as np
import cv2 as cv
import pyautogui
from enum import Enum
from os import listdir
from os.path import isdir, join, exists
from time import sleep, time
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF
from pysat.solvers import Solver
from itertools import compress
import traceback
import sys


id_pool = IDPool()
sixty_deg_rot = np.array([[np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60))],
                         [np.sin(np.deg2rad(60)), np.cos(np.deg2rad(60))]])


def locate_window():
    box = pyautogui.locateOnScreen('REMAINING.png')
    if box is None:
        raise RuntimeError('Could not locate Hexcells\' "REMAINING" lettering '
                           'on the screen and therefore could not find the Hexcells window!')
    x1, y1 = box.left - 1290, box.top - 20
    x2, y2 = x1 + 1440, y1 + 900
    print(f'Found window at {x1}, {y1}')
    return x1, y1, x2, y2


def crop_square(img, c, w):
    c = tuple(c.astype(int))
    l = w // 2
    u = l + w % 2
    return img[c[1] - l:c[1] + u, c[0] - l:c[0] + u]


def contour_center(contour):
    return np.mean(contour, axis=0).squeeze().round().astype(int)


def show_img(img):
    cv.imshow('f', img)
    while True:
        if cv.waitKey() == ord('q'):
            break
    cv.destroyWindow('f')


def encode_constraint(
        number,  # (int) number of hexagons in influence which should be blue
        curly,  # (bool) is curly
        dashed,  # (bool) is dashed
        influence,  # (List[Hexagon]) list of hexagons which are affected by the constraint
        circle  # (bool) if influenced hexagons are in a circle
):
    assert not circle or len(influence) <= 6
    assert len(influence) >= number

    # extract boolean variables
    literals = [h.var_id for h in influence]

    # if just a number
    if not curly and not dashed:
        return CardEnc.equals(lits=literals, bound=number, vpool=id_pool)

    # if dashed or curly
    # encode {number} in dnf
    n_vars = len(influence)
    dnf = []

    # this case needs special considerations since for hexagons gaps to count as such
    if (dashed or curly) and circle and n_vars < 6:
        # first find partitions
        # find start
        # remember: influence is sorted
        start = None
        for i in range(n_vars):
            # if there is a gap between influence[i] and influence[i - 1]
            if influence[i - 1] not in influence[i].influence or influence[i - 1].distance(influence[i]) > 1:
                start = i
                break
        assert start is not None

        # rotate such that start is at the beginning of influence
        influence = influence[start:] + influence[:start]
        partitions = [[influence[0]]]
        for i in influence[1:]:
            # if there is no gap between i and partitions[-1][-1]
            if i in partitions[-1][-1].influence and partitions[-1][-1].distance(i) == 1:
                partitions[-1].append(i)
            else:
                partitions.append([i])

        # encode {number} in dnf
        all_not = [-h.var_id for h in influence]
        start = 0
        for p in partitions:
            if len(p) < number:
                start += len(p)
                continue
            for i in range((len(p) - number + 1)):
                formula = all_not.copy()
                for j in range(number):
                    formula[start + i + j] = -formula[start + i + j]
                dnf.append(formula)
            start += len(p)

    else:
        # encode {number} in dnf
        all_not = [-h.var_id for h in influence]
        for i in range(n_vars if circle else (n_vars - number + 1)):
            formula = all_not.copy()
            for j in range(number):
                formula[(i + j) % n_vars] = -formula[(i + j) % n_vars]
            dnf.append(formula)

    if dashed:
        # -number- == not {number} and number
        cnf = CardEnc.equals(lits=literals, bound=number, vpool=id_pool)
        cnf.extend([[-v for v in c] for c in dnf])  # not {number} with DeMorgan
        return cnf

    if curly:
        # dnf to cnf
        cnf = [[]]
        for c in dnf:
            aux_var = id_pool.id(id_pool.top + 1)
            cnf[0].append(aux_var)
            for v in c:
                cnf.append([-aux_var, v])
        return CNF(from_clauses=cnf)




class Template:

    def __init__(self, name, imgs_path):
        self.name = name
        self.imgs = [cv.imread(join(imgs_path, f)) for f in listdir(join(imgs_path))]

    def get_score(self, img):
        # return the best score that was achieved across all templates
        return np.max([cv.matchTemplate(img, t, cv.TM_CCORR_NORMED).flatten().max() for t in self.imgs])

    @staticmethod
    def match(img, templates, threshold=.95):
        # match img against all templates
        # return the template with the highest score if the score > threshold
        s, t = max([(t.get_score(img), t.name) for t in templates], key=lambda x: x[0])
        return t if s > threshold else None  # t is the folder name e. g. "3cd" (3, center, dashed)

    @staticmethod
    def load_templates(name):
        path = join('templates', name)
        if not exists(path):
            return None
        return [Template(t, join(path, t)) for t in listdir(path) if isdir(join(path, t))]


BLUE_TEMPLATES = Template.load_templates('blue')
DARK_TEMPLATES = Template.load_templates('dark')
LINE_TEMPLATES = Template.load_templates('line')
LINE_TEMPLATES = [[x for x in LINE_TEMPLATES if x.name[1] == k or ('0' <= x.name[1] <= '9' and x.name[2] == k)] for k in ['l', 'c', 'r']]


class HexagonTypes(Enum):
    ORANGE = (False,)
    BLUE = (False,)
    BLUE_W_NUMBER = (True,)
    DARK = (True,)

    def __new__(cls, has_constraint):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, has_constraint):
        self.has_constraint = has_constraint


class Hexagon:
    def __init__(self, img, contour):
        self.contour = contour  # contour as returned by cv2.findContours
        self.center = contour_center(contour)  # center of the hexagon
        self.a = (np.max(self.contour[..., 0]) - np.min(self.contour[..., 0])) / 2  # side length
        self.influence = None  # list of hexagons that are affected by this hexagons constraints
        self.var_id = id_pool.id(id_pool.top + 1)  # boolean variable which represents this hexagon
        self.constraint = None  # boolean formula which encodes this hexagons constraints
        self.tpe = self._detect_type(img)  # HexagonType i. e. if blue, blue w/ number, orange, or gray
        self.other_hexagons = None  # list of all other hexagons
        self.detection = None  # detection as returned by Template.match

    def finish_init(self, img):
        if self.tpe.has_constraint:
            self._compute_constraints(img)

    def compute_influence(self, other_hexagons=None):

        if other_hexagons is not None:
            self.other_hexagons = other_hexagons
        assert self.other_hexagons is not None
        self.influence = []

        # compute influenced hexagons
        dst = 2 if self.tpe == HexagonTypes.BLUE_W_NUMBER else 1
        for hp in self.other_hexagons:
            if self.distance(hp) <= dst and hp is not self:
                self.influence.append(hp)

        assert not self.tpe == HexagonTypes.DARK or len(self.influence) <= 6
        assert not self.tpe == HexagonTypes.BLUE_W_NUMBER or len(self.influence) <= 18

        if self.tpe not in [HexagonTypes.BLUE, HexagonTypes.BLUE_W_NUMBER]:
            # sort influence
            a = sorted((h for h in self.influence if h.center[0] >= self.center[0] - self.a), key=lambda h: h.center[1])
            b = sorted((h for h in self.influence if h.center[0] < self.center[0] - self.a), key=lambda h: -h.center[1])
            assert len(self.influence) == len(a) + len(b)
            assert len(set(a) & set(b)) == 0
            self.influence = a + b

    def update(self, img):

        assert self.tpe is not None

        # if orange
        if np.min(np.max(np.abs(crop_square(img, self.center, 29) - (41, 176, 254)), axis=2)) <= 3:
            return False

        # detect new type (HexagonTypes)
        old_tpe = self.tpe
        self.tpe = self._detect_type(img)
        assert self.tpe != old_tpe

        # BLUE_W_NUMBER has larger influence
        if self.tpe == HexagonTypes.BLUE_W_NUMBER:
            self.compute_influence()

        # compute constraints
        if self.tpe.has_constraint:
            assert not old_tpe.has_constraint
            self._compute_constraints(img)
        return True

    def distance(self, other):
        # distance to other hexagon
        return max((np.linalg.norm(other.center - self.center) / self.a - .5), 0) // 2 + 1

    def _compute_constraints(self, img):

        number, curly, dashed = self._detect_constraint(img)

        if number == 'Q':  # question mark
            return
        circle = self.tpe == HexagonTypes.DARK  # if influenced hexagons are in a circle
        self.constraint = encode_constraint(int(number), curly, dashed, self.influence, circle)

    def _detect_type(self, img):
        # compute and return this hexagons type (HexagonTypes)
        c = self.center
        for i in range(6):
            indicator = img[c[1] - 10 - i, c[0]]
            if (indicator == 62).all():
                return HexagonTypes.DARK
            if (indicator == (41, 177, 255)).all():
                return HexagonTypes.ORANGE
            if (indicator == (235, 164, 5)).all():
                if np.mean(crop_square(img, c, 5) == (235, 164, 5)) > .8:
                    return HexagonTypes.BLUE
                else:
                    return HexagonTypes.BLUE_W_NUMBER
        # i = crop_square(img, c, 41).copy()
        # i[4:11, 20] = [0, 0, 255]
        # show_img(i)
        raise RuntimeError(f'Unknown hexagon type at {self.center} with color {indicator}')

    def _detect_constraint(self, img):
        assert self.tpe.has_constraint

        # detect number and is_curly or is_dashed

        template = DARK_TEMPLATES if self.tpe == HexagonTypes.DARK else BLUE_TEMPLATES
        det = Template.match(crop_square(img, self.center, 30), template)
        self.detection = det

        if det is None:
            raise RuntimeError(f'Couldn\'t match hexagon at {self.center}')

        t = 1 if len(det) > 1 and '0' <= det[1] <= '9' else 0
        number = det[:1+t]
        curly = det[1+t] == 'c' if len(det) > 1+t else False
        dashed = det[1+t] == 'd' if len(det) > 1+t else False
        return number, curly, dashed

    def check_if_constraint_trivial(self):
        # remove constraint if tautology
        if self.constraint is not None and all(h.tpe != HexagonTypes.ORANGE for h in self.influence):
            self.constraint = None


class Line:

    def __init__(self, img, hexagons, p, d, r):
        self.pos = p
        self.influence = None
        self.constraint = None
        self.detection = None
        self._compute_influence(hexagons, p, d, r)
        self._compute_constraint(img, r)

    def _compute_influence(self, hexagons, p, d, r):
        self.influence = []

        # compute influenced hexagons
        n = np.array([-d[1], d[0]])
        n = n / np.linalg.norm(n)
        c = np.matmul(n, p)
        for h in hexagons:
            if np.abs(np.matmul(n, h.center) - c) < h.a and \
                    ((r == 0 and np.all(h.center > p)) or
                     (r == 1 and h.center[1] > p[1]) or
                     (r == 2 and np.all((h.center > p) == [False, True]))):
                self.influence.append(h)
        self.influence.sort(key=lambda h: h.center[0] if r == 0 else h.center[1] if r == 1 else -h.center[0])

    def _compute_constraint(self, img, r):
        det = Template.match(crop_square(img, self.pos, 35), LINE_TEMPLATES[r])
        self.detection = det
        if det is None:
            raise RuntimeError(f'Couldn\'t match line at {self.pos}')
        t = 1 if '0' <= det[1] <= '9' else 0
        number = int(det[0:1+t])
        curly = det[2 + t] == 'c' if len(det) >= 3 + t else False
        dashed = det[2 + t] == 'd' if len(det) >= 3 + t else False
        assert len(self.influence) >= number
        self.constraint = encode_constraint(number, curly, dashed, self.influence, False)

    def check_if_constraint_trivial(self):
        if self.constraint is not None and all(h.tpe != HexagonTypes.ORANGE for h in self.influence):
            self.constraint = None
            # pyautogui.rightClick(*tuple(self.pos + (480, 56)))


class Game:
    def __init__(self):
        self.hexagons = []
        self.lines = []
        self.total_n_blue = None
        self.n_blue = None
        self.window_rect = locate_window()
        self._parse(self.take_screenshot())

    def take_screenshot(self):
        print('screenshot')
        pyautogui.moveTo((self.window_rect[0], self.window_rect[1] + 100))
        sleep(.05)
        i = np.asarray(pyautogui.screenshot().crop(self.window_rect))
        i = cv.cvtColor(i, cv.COLOR_RGB2BGR)
        # show_img(i)
        return i

    def debug_draw(self):
        img = self.take_screenshot()
        cv.drawContours(img, [h.contour for h in self.hexagons], -1, (0, 255, 0), 1)
        for h in self.hexagons:
            # if h.tpe != HexagonTypes.BLUE_W_NUMBER:
            #     continue
            c = h.center
            # cv.putText(img, str(h.var_id), (c[0] + 5, c[1]-10), cv.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 200), 1)
            if h.tpe.has_constraint:
                for n in h.influence:
                    cv.line(img, tuple(c + (1, 1)), tuple(n.center + (1, 1)), (0, 255, 0), 1)
                cv.putText(img, h.detection, (c[0]+5, c[1]+5), cv.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 200), 1)

        for l in self.lines:
            c = l.pos
            cv.line(img, tuple(c), tuple(l.influence[0].center), (0, 0, 0), 1)
            for i in range(len(l.influence) - 1):
                cv.line(img, tuple(l.influence[i].center), tuple(l.influence[i+1].center), (0, 0, 0), 1)
            cv.putText(img, l.detection, (c[0]+5, c[1]), cv.FONT_HERSHEY_SIMPLEX, .5, (200, 0, 200), 1)
            # for i in l.influence:
            #     cv.putText(img, 'l', (i.center[0] - 5, i.center[1] + 5), cv.FONT_HERSHEY_SIMPLEX, .5, (150, 0, 150), 1)

        return img

    def solve(self):
        n_orange = len([h for h in self.hexagons if h.tpe == HexagonTypes.ORANGE])
        use_total = False
        while True:
            with Solver() as s:

                if use_total:
                    assert self.total_n_blue is not None, 'Need total number of blue to solve'
                    is_orange = [h.var_id for h in self.hexagons if h.tpe == HexagonTypes.ORANGE]
                    s.append_formula(CardEnc.equals(is_orange, self.total_n_blue - self.n_blue, vpool=id_pool).clauses)
                    print('Total active')

                assumptions = []
                formulas = []
                for h in self.hexagons:
                    formulas.append((h.var_id, h.constraint))
                    if h.constraint is not None:
                        s.append_formula(h.constraint)
                    if h.tpe == HexagonTypes.DARK:
                        assumptions.append(-h.var_id)
                    elif h.tpe in [HexagonTypes.BLUE, HexagonTypes.BLUE_W_NUMBER]:
                        assumptions.append(h.var_id)
                    # s.solve(assumptions=assumptions)

                for l in self.lines:
                    formulas.append((l.detection, h.constraint))
                    if l.constraint is not None:
                        s.append_formula(l.constraint)
                        # assert s.solve(assumptions=assumptions)

                if not s.solve(assumptions=assumptions):
                    raise RuntimeError('The boolean formula is unsatisfiable. This usually occurs when a game element is detected wrongly.')

                hexagons_to_update = []

                for h in reversed(self.hexagons):
                    if h.tpe != HexagonTypes.ORANGE:
                        continue
                    if not s.solve(assumptions=assumptions + [h.var_id]):
                        print(-h.var_id)
                        pyautogui.rightClick(*tuple(h.center + self.window_rect[:2]))
                        hexagons_to_update.append(h)
                    elif not s.solve(assumptions=assumptions + [-h.var_id]):
                        print(h.var_id)
                        pyautogui.leftClick(*tuple(h.center + self.window_rect[:2]))
                        hexagons_to_update.append(h)
                        self.n_blue += 1

                n_orange -= len(hexagons_to_update)
                if n_orange == 0:
                    break
                if self.total_n_blue is not None:
                    print('REMAINING', self.total_n_blue - self.n_blue)

                if len(hexagons_to_update) == 0:
                    if use_total:
                        raise RuntimeError('Can\'t solve the puzzle! Solution is ambiguous.')
                    use_total = True
                    continue

                sleep(.3)

                start_time = time()

                while len(hexagons_to_update) > 0:
                    img = self.take_screenshot()
                    failed = []
                    for h in hexagons_to_update:
                        print('updating', h.var_id)
                        failed.append(not h.update(img))
                    hexagons_to_update = list(compress(hexagons_to_update, failed))
                    if time() - start_time >= 3:
                        raise RuntimeError('Failed to update state of game elements')

                for x in self.hexagons + self.lines:
                    x.check_if_constraint_trivial()

    def _parse(self, img):
        print('Detecting game elements')
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 200, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        self.hexagons = [Hexagon(img, c) for c in contours if 1500 < cv.contourArea(c) < 1700]
        for h in self.hexagons:
            h.compute_influence(self.hexagons)
        for h in self.hexagons:
            h.finish_init(img)

        self.n_blue = len([h for h in self.hexagons if h.tpe == HexagonTypes.BLUE or h.tpe == HexagonTypes.BLUE_W_NUMBER])

        if self.total_n_blue is None:
            # searching for the blue box which shows the number of remaining blue hexagons
            cc = [c for c in contours if 8000 < cv.contourArea(c) < 12000]
            box = min(cc, key=lambda c: np.min(c))
            box_center = contour_center(box)

            i = img[box_center[1] - 30:box_center[1] + 30, box_center[0] - 85:box_center[0] + 85]
            n_blue_templates = Template.load_templates('n_blue')

            # detect the number of remaining blue hexagons with template matching
            det = Template.match(i, n_blue_templates)
            if det is not None:
                self.total_n_blue = int(det) + self.n_blue
                print(f'total_n_blue: {self.total_n_blue} ({int(det)})')
            else:
                print('Warning: Template matching failed, did not detect the total number of hexagons that should be blue')

        line_candidates = []
        for h in self.hexagons:
            if h.tpe is HexagonTypes.DARK and len(h.influence) == 6:
                continue
            d = 1.5 * h.a * np.array([[-0.866], [-0.5]])
            for r in range(3):
                p = (h.center + d.squeeze()).astype(int)
                for hp in self.hexagons:
                    if np.linalg.norm((p - hp.center)) < hp.a:
                        break
                else:
                    numberness = np.sum(crop_square(img, p, 10).mean(axis=2) < 230)
                    if numberness > 0:

                        same_pos = [(i, l) for i, l in enumerate(line_candidates) if np.linalg.norm(l[0] - p) < h.a * .6]
                        assert len(same_pos) <= 1
                        if len(same_pos) == 0 or same_pos[0][1][3] < numberness:
                            if len(same_pos) == 1:
                                del line_candidates[same_pos[0][0]]
                            line_candidates.append((p, d, r, numberness))

                d = np.matmul(sixty_deg_rot, d)
                # cv.circle(self.img, tuple(p), int(h.a // 2), (0, 255, 0), 1)

        for l in line_candidates:
            self.lines.append(Line(img, self.hexagons, l[0], -l[1].squeeze(), l[2]))


if __name__ == '__main__':
    # resolution 1440x900
    # font: https://fontsme.com/wp-data/h/487/14487/file/Harabara Mais.otf

    pyautogui.PAUSE = .001  # sleep in seconds after each mouse click

    # wait for the user to bring the Hexcells window to foreground
    print(f'Sleeping {pyautogui.PAUSE}s after each mouse click')
    print('Please bring the Hexcells window to front')
    print('Make sure it has a resolution of 1440x900!')
    for i in range(10):
        sleep(1)
        print(f'Searching for Hexcells window {i+1}/10')
        if pyautogui.locateOnScreen('REMAINING.png') is not None:
            break
    else:
        print('Error: Could not locate Hexcells\' "REMAINING" lettering on the screen '
              'and therefore could not find the Hexcells window!', file=sys.stderr)
        exit()

    # parse and solve the game
    g = Game()
    try:
        g.solve()
    except RuntimeError and AssertionError as e:
        traceback.print_exc()
        show_img(g.debug_draw())
